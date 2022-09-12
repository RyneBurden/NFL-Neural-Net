import os
import time

for x in range(1, 19):
    os.system(f"Rscript.exe staley_driver.R 2021 {x} TRUE")
    time.sleep(5)
