#!/bin/bash
if [ "$1" = "run" ]
then
	echo "jupyter notebook start"
	nohup jupyter notebook & 
elif [ "$1" = "stop" ]
then
	pkill -9 -ef jupyter-notebook
	echo "jupyter notebook stop"
else
	echo "insert args: run / stop"
fi