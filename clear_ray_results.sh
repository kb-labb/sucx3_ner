#!/usr/bin/bash
for i in $(find ray_results/ -type f| \grep -v "tfevents"); do rm -f $i; done
find ray_results/ -type d -empty -delete