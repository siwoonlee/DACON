# Sentence Classification Project

## How to run?
1. Install Docker on your computer
2. Run docker command
   - docker build --tag sentence:1.0 . 
   - docker run -itd --ipc=host --net=host -v [HOST SHARE VOLUME DIR]:[CONTAINER VOLUME DIR] --gpus all sentence:1.0
     - HOST SHARE VOLUME DIR - it should be current absolute directory path
     - CONTAINER VOLUME DIR - /home should be fine
3. Go into docker container by following command
   - docker ps
   - docker exec -it [CONTAINER_ID] /bin/sh
4. Run python scripts
   - python train.py
