docker run -it -v /mnt:/mnt -v /home:/home -v $PWD/../:/root/workdir -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=":0.0" --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  --env QT_X11_NO_MITSHM=1 --privileged   labelpose --pose true
