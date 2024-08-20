FROM nvidia/cuda:12.2.0-base-ubuntu22.04
WORKDIR /app
COPY . .
RUN apt update && \
  apt-get install -y --no-install-recommends \
           make \
           git \
           libgl1-mesa-glx \
      && apt-get clean && \
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

