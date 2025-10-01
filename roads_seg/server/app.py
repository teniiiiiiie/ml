"""FastAPI application for image and video segmentation using Triton inference server."""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    FastAPI = None
    File = None
    UploadFile = None
    HTTPException = None
    JSONResponse = None

try:
    import tritonclient.grpc as grpcclient
except ImportError:
    grpcclient = None

import numpy as np
import cv2  # pylint: disable=import-error
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI() if FastAPI else None

TRITON_HOST = os.getenv("TRITON_HOST", "triton-server")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")
MODEL_NAME = os.getenv("MODEL_NAME", "roads")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "images")

TRITON_CLIENT: Optional[Any] = None
try:
    if grpcclient:
        TRITON_CLIENT = grpcclient.InferenceServerClient(url=f"{TRITON_HOST}:{TRITON_PORT}")
        logger.info("Triton client initialized")
except Exception as exc:  # pylint: disable=broad-exception-caught
    logger.error("Failed to initialize Triton client: %s", exc)

S3_CLIENT: Optional[Any] = None
try:
    S3_CLIENT = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        verify=False,
    )
    logger.info("S3 client initialized")
    try:
        S3_CLIENT.head_bucket(Bucket=BUCKET_NAME)
        logger.info("Bucket %s exists", BUCKET_NAME)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            S3_CLIENT.create_bucket(Bucket=BUCKET_NAME)
            logger.info("Bucket %s created", BUCKET_NAME)
        else:
            logger.warning("Bucket issue: %s", exc)
except Exception as exc:  # pylint: disable=broad-exception-caught
    logger.error("Failed to initialize S3 client: %s", exc)


def get_model_metadata() -> Optional[Any]:
    """Fetch model metadata from Triton server."""
    if TRITON_CLIENT is None:
        return None
    try:
        model_metadata = TRITON_CLIENT.get_model_metadata(model_name=MODEL_NAME)  # pylint: disable=no-member
        input_names = [input_obj.name for input_obj in model_metadata.inputs]
        output_names = [output_obj.name for output_obj in model_metadata.outputs]
        logger.info("Model inputs: %s", input_names)
        logger.info("Model outputs: %s", output_names)
        return model_metadata
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Failed to get model metadata: %s", exc)
        return None


get_model_metadata()


def check_services() -> List[str]:
    """Check availability of required services."""
    issues: List[str] = []
    if not TRITON_CLIENT:
        issues.append("Triton client not initialized")
    else:
        try:
            if not TRITON_CLIENT.is_server_live():  # pylint: disable=no-member
                issues.append("Triton server not live")
            if not TRITON_CLIENT.is_model_ready(model_name=MODEL_NAME):  # pylint: disable=no-member
                issues.append(f"Model {MODEL_NAME} not ready")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            issues.append(f"Triton connection error: {exc}")
    if not S3_CLIENT:
        issues.append("S3 client not initialized")
    else:
        try:
            S3_CLIENT.list_buckets()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            issues.append(f"S3 connection error: {exc}")
    return issues


def process_image_with_triton(image_data: np.ndarray) -> List[List[int]]:
    """Process image through Triton inference server."""
    if TRITON_CLIENT is None:
        raise HTTPException(status_code=500, detail="Triton client not available")  # pylint: disable=used-before-assignment
    try:
        metadata = TRITON_CLIENT.get_model_metadata(model_name=MODEL_NAME)  # pylint: disable=no-member
        if metadata is None:
            raise HTTPException(status_code=500, detail="Failed to get model metadata")  # pylint: disable=used-before-assignment

        in_name = metadata.inputs[0].name
        out_name = metadata.outputs[0].name

        logger.info("Using input: %s, output: %s", in_name, out_name)

        rgb_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member

        orig_h, orig_w = rgb_img.shape[:2]
        tgt_size = (512, 512)

        resized = cv2.resize(rgb_img, tgt_size, interpolation=cv2.INTER_LINEAR)  # pylint: disable=no-member

        arr = resized.astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        arr = np.transpose(arr, (0, 3, 1, 2))

        logger.info("Input shape after preprocessing: %s", arr.shape)

        infer_in = [grpcclient.InferInput(in_name, arr.shape, "FP32")]  # pylint: disable=used-before-assignment
        infer_in[0].set_data_from_numpy(arr)
        infer_out = [grpcclient.InferRequestedOutput(out_name)]  # pylint: disable=used-before-assignment

        res = TRITON_CLIENT.infer(MODEL_NAME, infer_in, outputs=infer_out)  # pylint: disable=no-member
        out_arr = res.as_numpy(out_name)

        logger.info("Output shape: %s", out_arr.shape)

        if out_arr.shape[1] > 1:
            msk = np.argmax(out_arr[0], axis=0).astype(np.uint8)
        else:
            msk = (out_arr[0, 0] > 0.5).astype(np.uint8)

        msk_resized = cv2.resize(  # pylint: disable=no-member
            msk, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST  # pylint: disable=no-member
        )

        return msk_resized.tolist()

    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}") from exc  # pylint: disable=used-before-assignment


def save_to_s3(key: str, data: Union[Dict[Any, Any], List[Any], str]) -> None:
    """Save data to S3 storage."""
    if S3_CLIENT is None:
        raise HTTPException(status_code=500, detail="S3 client not available")  # pylint: disable=used-before-assignment
    try:
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        S3_CLIENT.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=data_str.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("Successfully saved %s to S3", key)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Failed to save to S3: %s", exc)
        raise HTTPException(status_code=500, detail=f"S3 save error: {str(exc)}") from exc  # pylint: disable=used-before-assignment


@app.get("/health") if app else None
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    issues = check_services()
    if issues:
        return {"status": "unhealthy", "issues": issues}
    return {"status": "healthy"}


@app.post("/segment") if app else None
async def segment_image(file: UploadFile = File(...)) -> JSONResponse:  # pylint: disable=used-before-assignment
    """Segment image and return mask."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(  # pylint: disable=used-before-assignment
            status_code=400, detail="Invalid file type. Please upload an image."
        )
    try:
        img_bytes = await file.read()
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img_data = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)  # pylint: disable=no-member

        if img_data is None:
            raise HTTPException(status_code=400, detail="Invalid image file")  # pylint: disable=used-before-assignment

        orig_h, orig_w = img_data.shape[:2]
        res = process_image_with_triton(img_data)

        if S3_CLIENT is None:
            raise HTTPException(status_code=500, detail="S3 client not available")  # pylint: disable=used-before-assignment

        S3_CLIENT.put_object(
            Bucket=BUCKET_NAME,
            Key=f"original_{file.filename}",
            Body=img_bytes,
            ContentType=file.content_type,
        )

        save_to_s3(f"mask_{file.filename}.json", {"mask": res})

        msk_shape = (
            [len(res), len(res[0])] if res and isinstance(res[0], list) else [0, 0]
        )

        resp = {
            "message": "Image segmented successfully",
            "mask_shape": msk_shape,
            "original_shape": [orig_h, orig_w],
            "saved_keys": [f"original_{file.filename}", f"mask_{file.filename}.json"],
        }

        return JSONResponse(content=resp)  # pylint: disable=used-before-assignment

    except HTTPException:  # pylint: disable=used-before-assignment
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Image segmentation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(exc)}") from exc  # pylint: disable=used-before-assignment


@app.post("/segment_video") if app else None
async def segment_video(data: Dict[str, Any]) -> JSONResponse:  # pylint: disable=used-before-assignment
    """Segment video and return masks."""
    vid_key = data.get("video_key")
    if not vid_key:
        raise HTTPException(status_code=400, detail="video_key is required")  # pylint: disable=used-before-assignment

    try:
        if S3_CLIENT is None:
            raise HTTPException(status_code=500, detail="S3 client not available")  # pylint: disable=used-before-assignment
        resp = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=vid_key)
        vid_data = resp["Body"].read()

        tmp_path = "/tmp/temp_video.mp4"
        with open(tmp_path, "wb") as f:
            f.write(vid_data)

        cap = cv2.VideoCapture(tmp_path)  # pylint: disable=no-member
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")  # pylint: disable=used-before-assignment

        fps_val = cap.get(cv2.CAP_PROP_FPS)  # pylint: disable=no-member
        skip = max(1, int(fps_val / 6))
        masks: List[Dict[str, Any]] = []
        cnt = 0

        while True:
            ret_val, frm = cap.read()
            if not ret_val:
                break

            if cnt % skip == 0:
                msk_data = process_image_with_triton(frm)
                masks.append(
                    {
                        "frame": cnt,
                        "timestamp": cnt / fps_val if fps_val > 0 else 0,
                        "mask": msk_data,
                    }
                )

            cnt += 1

        cap.release()

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        save_to_s3(f"video_masks_{vid_key}.json", masks)

        resp_data = {
            "message": "Video segmented successfully",
            "frames_processed": len(masks),
            "total_frames": cnt,
            "result_key": f"video_masks_{vid_key}.json",
        }

        return JSONResponse(content=resp_data)  # pylint: disable=used-before-assignment

    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Video not found in S3") from exc  # pylint: disable=used-before-assignment
        raise HTTPException(status_code=500, detail=f"S3 error: {str(exc)}") from exc  # pylint: disable=used-before-assignment
    except HTTPException:  # pylint: disable=used-before-assignment
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Video segmentation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(exc)}") from exc  # pylint: disable=used-before-assignment


@app.get("/list_files") if app else None
async def list_files() -> Dict[str, Any]:
    """List files in S3 bucket."""
    if S3_CLIENT is None:
        raise HTTPException(status_code=500, detail="S3 client not available")  # pylint: disable=used-before-assignment
    try:
        resp = S3_CLIENT.list_objects_v2(Bucket=BUCKET_NAME)
        files = [obj["Key"] for obj in resp.get("Contents", [])]
        return {"files": files}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise HTTPException(status_code=500, detail=f"S3 list error: {str(exc)}") from exc  # pylint: disable=used-before-assignment


if __name__ == "__main__":
    try:
        import uvicorn
        if app:
            uvicorn.run(app, host="0.0.0.0", port=80)
    except ImportError:
        pass
