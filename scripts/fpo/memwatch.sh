#!/bin/bash
# log cgroup usage and top RSS processes every 20s until file stop exists
while [ ! -e stop_memwatch ]; do
  echo "$(date +%T) cg=$(( $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)/1000000 ))MB rss_cg=$(awk '/^(total_)?rss /{print int($2/1000000)}' /sys/fs/cgroup/memory/memory.stat | head -1)MB cache=$(awk '/^cache /{print int($2/1000000)}' /sys/fs/cgroup/memory/memory.stat)MB" >> memwatch.log
  ps -eo rss,args --sort=-rss | head -5 | awk '{printf "   %d MB %s %s\n",$1/1024,$2,$3}' >> memwatch.log
  sleep 20
done
