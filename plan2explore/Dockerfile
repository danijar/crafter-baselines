FROM tensorflow/tensorflow:2.6.0-gpu

RUN apt-get update && apt-get install -y \
  python3-pip \
  && apt-get clean

RUN pip3 install --no-cache-dir \
  crafter>=1.6.2 \
  dreamerv2>=2.0.7 \
  gym

COPY . /app
WORKDIR /app
CMD ["python3", "main.py", "--logdir", "/logdir"]
