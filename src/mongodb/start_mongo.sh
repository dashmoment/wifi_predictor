#! /bin/sh
#--fork: start mongodb as daemon
ulimit -n 6204
mongod --fork --config /etc/mongod.conf

