#!/bin/bash

experiments="nonconvex_cost;quadratic_utility constraint;bottom constraint;top non_inc_util_non_convex_cost;noninc_util non_inc_util_non_convex_cost;non_convex_cost constraint_realistic;bottom"

for ex in $experiments ; do 
  plot_type=`echo $ex | cut -d";" -f1`
  plot_type_suboption=`echo $ex | cut -d";" -f2`
  echo "python main.py $plot_type $plot_type_suboption donotshowfigs" 
  python main.py $plot_type $plot_type_suboption donotshowfigs;
done

