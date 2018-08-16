# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:27:13 2017

@author: MaggieYC_Pang
"""

"""
A small example subscriber
"""
import paho.mqtt.client as paho
import struct
import json

def decode_stainfo(payload):
    pos = 0
    mac_addr_ap = struct.unpack_from("6B", payload, pos)
    print( "mac_addr_ap = %02x:%02x:%02x:%02x:%02x:%02x" % (mac_addr_ap[0],mac_addr_ap[1],mac_addr_ap[2],mac_addr_ap[3],mac_addr_ap[4],mac_addr_ap[5]))
    pos += 6
    
    (sec,minute,hr,day,mon,yr,usec) = struct.unpack_from(">iiiiiii", payload, pos)
    pos += 4*7
    
    mac_addr_sta = struct.unpack_from("6B", payload, pos)
    print( "mac_addr_sta = %02x:%02x:%02x:%02x:%02x:%02x" % (mac_addr_sta[0],mac_addr_sta[1],mac_addr_sta[2],mac_addr_sta[3],mac_addr_sta[4],mac_addr_sta[5]))
    pos += 6
    
    (inactive_time, rx_bytes, rx_packets, tx_bytes, tx_packets) = struct.unpack_from(">iIIII", payload, pos)
    pos += 5*4

    (tx_retries, tx_failed, connected_time, signal) = struct.unpack_from(">IIIi", payload, pos)
    pos += 4*4
	
    (tx_bitrate, rx_bitrate, expected_tput) = struct.unpack_from(">ddd", payload, pos)
    pos += 8*3
    
    print( "last_update = %d/%d/%d %02d:%02d:%02d.%06d" % (yr,mon,day,hr,minute,sec,usec))
    print( "inactive_time = %d ms" % inactive_time)
    print( "connected_time = %d sec" % connected_time)
    print( "rx_bytes = %d" % rx_bytes)
    print( "rx_packets = %d" % rx_packets)
    print( "tx_bytes = %d" % tx_bytes)
    print( "tx_packets = %d" % tx_packets)
    print( "tx_retries = %d" % tx_retries)
    print( "tx_failed = %d" % tx_failed)
    print( "signal = %d" % signal)
    print( "tx_bitrate = %f" % tx_bitrate)
    print( "rx_bitrate = %f" % rx_bitrate)
    print( "expected_tput = %f" % expected_tput)
    print( "")
    

def decode_devsurvey(payload):
    pos = 0

    mac_addr_dev = struct.unpack_from("6B", payload, pos)
    print( "mac_addr_dev = %02x:%02x:%02x:%02x:%02x:%02x" % (mac_addr_dev[0],mac_addr_dev[1],mac_addr_dev[2],mac_addr_dev[3],mac_addr_dev[4],mac_addr_dev[5]))
    pos += 6
    
    (sec,minute,hr,day,mon,yr,usec) = struct.unpack_from(">iiiiiii", payload, pos)
    pos += 4*7
    
    (freq, noise, active_time, busy_time, receive_time, transmit_time) = struct.unpack_from(">IBQQQQ", payload, pos)
    pos += 4 + 1 + 4*8
    
#    print( "last_update = %d/%d/%d %02d:%02d:%02d.%06d" % (yr,mon,day,hr,minute,sec,usec))
    print( "freq = %d MHz" % freq)
#    print( "noise = %d dBm" % noise)
#    print( "active_time = %d ms" % active_time)
    print( "busy_time = %d ms" % busy_time)
#    print( "receive_time = %d ms" % receive_time)
#    print( "transmit_time = %d ms" % transmit_time)
    print( "")

    
def on_message(mosq, obj, msg):
    print( "%-20s %d" % (msg.topic, msg.length))
    print( type(msg.payload))
#    print( "%s" % (msg.payload.decode("UTF-8")))
#    if(msg.topic == "cwmdata/stainfo_update"):
#        decode_stainfo(msg.payload)
    if(msg.topic == "CwmData/DevSurveyUpdate"):
#        decode_devsurvey(msg.payload)
#        print(json.dumps(msg.payload.decode("UTF-8"), "indent=4"))
        decodejson =  json.loads(msg.payload.decode("latin-1"))
#        decodejson =  json.loads( \' + encodedjson + \' )
#        print( "%s : %s" % (type(decodejson), decodejson))
#        print( "%s" % decodejson)
        for key, value in decodejson.items():
            print( key, value )
#            if(key == "Payload"):
#                decode_devsurvey(value.encode("latin-1"))
    mosq.publish('pong', 'ack', 0)

def on_publish(mosq, obj, mid):
    pass
    
if __name__ == '__main__':
    client = paho.Client()
    client.on_message = on_message
    client.on_publish = on_publish

    #client.tls_set('root.ca', certfile='c1.crt', keyfile='c1.key')
    client.connect("10.144.24.149", 1234, 60)
    #client.connect("127.0.0.1", 1883, 60)

    client.subscribe("CwmData/#", 0)


    while client.loop() == 0:
        pass

# vi: set fileencoding=utf-8 :