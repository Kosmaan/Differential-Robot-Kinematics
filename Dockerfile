FROM osrf/ros:humble-desktop-full

RUN apt update
RUN apt upgrade -y
#RUN apt install -y ros-humble-geometry-msgs\
#		ros-humble-sensor-msgs\
#		python3-pip
#RUN pip install numpy
RUN useradd -ms /bin/bash student

USER student
WORKDIR /home/student/scr_ws

COPY . .
RUN colcon build
#RUN echo "source /home/student/scr_ws/install/setup.bash" >> /.bashrc
