#!/bin/sh
$HOME/anaconda3/envs/vimh/bin/gunicorn -w 4 server:app --bind localhost:9000