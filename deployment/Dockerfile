FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir \
    torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers>=4.40.0 peft>=0.10.0 bitsandbytes>=0.43.0 \
    accelerate>=0.28.0 boto3>=1.26.0

COPY inference/ ./inference/
COPY training/ ./training/

CMD ["inference.handler.handler"]
