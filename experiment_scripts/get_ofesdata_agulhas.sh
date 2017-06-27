#!/bin/bash
mkdir ofesdata
for i in {3165..3287}
do
  export out=`printf "%05d\n" $i`
  echo "getting files at time $i"
  ncks -v 'uvel' -d time,$i -d lon,50,351 -d lat,300,501  http://apdrc.soest.hawaii.edu:80/dods/public_ofes/OfES/ncep_0.1_global_3day/uvel -o ofesdata/uvel${out}.nc
  ncks -v 'vvel' -d time,$i -d lon,50,351 -d lat,300,501  http://apdrc.soest.hawaii.edu:80/dods/public_ofes/OfES/ncep_0.1_global_3day/vvel -o ofesdata/vvel${out}.nc
  ncks -v 'wvel' -d time,$i -d lon,50,351 -d lat,300,501  http://apdrc.soest.hawaii.edu:80/dods/public_ofes/OfES/ncep_0.1_global_3day/wvel -o ofesdata/wvel${out}.nc
  ncks -v 'temp' -d time,$i -d lon,50,351 -d lat,300,501  http://apdrc.soest.hawaii.edu:80/dods/public_ofes/OfES/ncep_0.1_global_3day/temp -o ofesdata/temp${out}.nc
done
