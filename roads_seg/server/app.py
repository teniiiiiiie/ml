"""FastAPI application for image and video segmentation using Triton inference server."""
import os
import json
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRITON_HOST = os.getenv("TRITON_HOST", "triton-server")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")
MODEL_NAME = os.getenv("MODEL_NAME", "roads")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "images")

TRITON_CLIENT: Optional[grpcclient.InferenceServerClient] = None
S3_CLIENT: Optional[Any] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global TRITON_CLIENT, S3_CLIENT
    
    logger.info("Initializing application services...")
    
    try:
        TRITON_CLIENT = grpcclient.InferenceServerClient(
            url=f"{TRITON_HOST}:{TRITON_PORT}"
        )
        logger.info("Triton client initialized successfully")
        
        max_retries = 30
        for attempt in range(max_retries):
            try:
                if TRITON_CLIENT.is_model_ready(model_name=MODEL_NAME):
                    logger.info(f"Model '{MODEL_NAME}' is ready")
                    break
                else:
                    logger.info(f"Waiting for model '{MODEL_NAME}' to be ready... (Attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.warning(f"Model check failed: {e}")
            
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
        else:
            logger.error(f"Model '{MODEL_NAME}' failed to become ready in time")
                
    except Exception as e:
        logger.error(f"Failed to initialize Triton client: {e}")

    try:
        S3_CLIENT = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            verify=False,
        )
        S3_CLIENT.head_bucket(Bucket=BUCKET_NAME)
        logger.info("S3 client initialized successfully")
        logger.info(f"Bucket '{BUCKET_NAME}' is accessible")
        
    except ClientError as e:
        logger.error(f"S3 bucket verification failed: {e}")
        S3_CLIENT = None
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        S3_CLIENT = None

    yield

    if TRITON_CLIENT:
        TRITON_CLIENT.close()
        logger.info("Triton client closed")

app = FastAPI(
    lifespan=lifespan,
    title="Segmentation API",
    description="API for image and video segmentation using Triton inference server",
    version="1.0.0"
)

def check_services() -> List[str]:
    issues: List[str] = []
    
    if not TRITON_CLIENT:
        issues.append("Triton client not initialized")
    else:
        try:
            if not TRITON_CLIENT.is_server_live():
                issues.append("Triton server not live")
            if not TRITON_CLIENT.is_model_ready(model_name=MODEL_NAME):
                issues.append(f"Model {MODEL_NAME} not ready")
        except Exception as e:
            issues.append(f"Triton connection error: {e}")
    
    if not S3_CLIENT:
        issues.append("S3 client not initialized")
    else:
        try:
            S3_CLIENT.list_buckets()
        except Exception as e:
            issues.append(f"S3 connection error: {e}")
    
    return issues

def get_model_metadata() -> Optional[Any]:
    if TRITON_CLIENT is None:
        return None
    try:
        model_metadata = TRITON_CLIENT.get_model_metadata(model_name=MODEL_NAME)
        input_names = [input_obj.name for input_obj in model_metadata.inputs]
        output_names = [output_obj.name for output_obj in model_metadata.outputs]
        logger.info(f"Model inputs: {input_names}")
        logger.info(f"Model outputs: {output_names}")
        return model_metadata
    except Exception as e:
        logger.error(f"Failed to get model metadata: {e}")
        return None

def process_image_with_triton(image_data: np.ndarray) -> List[List[int]]:
    if TRITON_CLIENT is None:
        raise HTTPException(status_code=500, detail="Triton client not available")
    
    try:
        metadata = TRITON_CLIENT.get_model_metadata(model_name=MODEL_NAME)
        if metadata is None:
            raise HTTPException(status_code=500, detail="Failed to get model metadata")

        in_name = metadata.inputs[0].name
        out_name = metadata.outputs[0].name

        logger.info(f"Using input: {in_name}, output: {out_name}")

        rgb_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb_img.shape[:2]
        tgt_size = (512, 512)

        resized = cv2.resize(rgb_img, tgt_size, interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        arr = np.transpose(arr, (0, 3, 1, 2))

        logger.info(f"Input shape after preprocessing: {arr.shape}")

        infer_in = [grpcclient.InferInput(in_name, arr.shape, "FP32")]
        infer_in[0].set_data_from_numpy(arr)
        infer_out = [grpcclient.InferRequestedOutput(out_name)]

        res = TRITON_CLIENT.infer(MODEL_NAME, infer_in, outputs=infer_out)
        out_arr = res.as_numpy(out_name)

        logger.info(f"Output shape: {out_arr.shape}")

        if out_arr.shape[1] > 1:
            msk = np.argmax(out_arr[0], axis=0).astype(np.uint8)
        else:
            msk = (out_arr[0, 0] > 0.5).astype(np.uint8)

        msk_resized = cv2.resize(
            msk, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )

        return msk_resized.tolist()

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}") from e

def save_to_s3(key: str, data: Union[Dict[Any, Any], List[Any], str]) -> None:
    if S3_CLIENT is None:
        raise HTTPException(status_code=500, detail="S3 client not available")
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
        logger.info(f"Successfully saved {key} to S3")
    except Exception as e:
        logger.error(f"Failed to save to S3: {e}")
        raise HTTPException(status_code=500, detail=f"S3 save error: {str(e)}") from e

@app.get("/")
async def root():
    return {
        "message": "Segmentation API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "segment_image": "/segment",
            "segment_video": "/segment_video",
            "list_files": "/list_files"
        }
    }

@app.get("/health")
async def health() -> Dict[str, Any]:
    issues = check_services()
    if issues:
        return {"status": "unhealthy", "issues": issues}
    return {"status": "healthy"}

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an image."
        )
    
    try:
        img_bytes = await file.read()
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img_data = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if img_data is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        orig_h, orig_w = img_data.shape[:2]
        res = process_image_with_triton(img_data)

        if S3_CLIENT is None:
            raise HTTPException(status_code=500, detail="S3 client not available")

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

        return JSONResponse(content=resp)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}") from e

@app.post("/segment_video")
async def segment_video(data: Dict[str, Any]) -> JSONResponse:
    vid_key = data.get("video_key")
    if not vid_key:
        raise HTTPException(status_code=400, detail="video_key is required")

    try:
        if S3_CLIENT is None:
            raise HTTPException(status_code=500, detail="S3 client not available")
        
        resp = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=vid_key)
        vid_data = resp["Body"].read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(vid_data)

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Cannot open video file")

            fps_val = cap.get(cv2.CAP_PROP_FPS)
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

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        save_to_s3(f"video_masks_{vid_key}.json", masks)

        resp_data = {
            "message": "Video segmented successfully",
            "frames_processed": len(masks),
            "total_frames": cnt,
            "result_key": f"video_masks_{vid_key}.json",
        }

        return JSONResponse(content=resp_data)

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Video not found in S3") from e
        raise HTTPException(status_code=500, detail=f"S3 error: {str(e)}") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}") from e

@app.get("/list_files")
async def list_files() -> Dict[str, Any]:
    if S3_CLIENT is None:
        raise HTTPException(status_code=500, detail="S3 client not available")
    try:
        resp = S3_CLIENT.list_objects_v2(Bucket=BUCKET_NAME)
        files = [obj["Key"] for obj in resp.get("Contents", [])]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 list error: {str(e)}") from e

@app.get("/model_info")
async def model_info():
    if TRITON_CLIENT is None:
        raise HTTPException(status_code=500, detail="Triton client not available")
    
    try:
        metadata = get_model_metadata()
        if metadata is None:
            raise HTTPException(status_code=500, detail="Failed to get model metadata")
        
        model_info = {
            "name": MODEL_NAME,
            "inputs": [
                {
                    "name": input_obj.name,
                    "datatype": input_obj.datatype,
                    "shape": list(input_obj.shape)
                }
                for input_obj in metadata.inputs
            ],
            "outputs": [
                {
                    "name": output_obj.name,
                    "datatype": output_obj.datatype,
                    "shape": list(output_obj.shape)
                }
                for output_obj in metadata.outputs
            ]
        }
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}") from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
