
import time
t1 = time.time()

# import rides_concat
# import passengers_concat

import week_report_math
import week_report_math_8
import week_report_math_driver
import week_report_math_route

#合并文件名new_file_rides.csv
#文件名new_file_passagers.csv
print("total_math完成！")
t2 = time.time()

mint = (t2-t1)/60
print("周报总运行时间",mint,"minutes")