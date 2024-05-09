import multiprocessing as mp
import rawpy
import cv2

def load_raw_from_bytesio(b):
    with rawpy.imread(b) as raw:
        # Access the raw image data
        raw_data = raw.raw_image
        height, width = raw_data.shape
        raw_array = raw_data.reshape((height, width, 1))
        return b, cv2.cvtColor(raw_array, 46)


def main():
    files = ["D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0462.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0463.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0464.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0465.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0466.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0467.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0468.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0469.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0470.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0471.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0472.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0473.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0474.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0475.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0476.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0477.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0478.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0479.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0480.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0481.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0482.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0483.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0484.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0485.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0486.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0487.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0488.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0489.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0490.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0491.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0492.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0493.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0494.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0495.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0496.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0497.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0498.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0499.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0500.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0501.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0502.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0503.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0504.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0505.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0506.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0507.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0508.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0509.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0510.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0511.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0512.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0513.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0514.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0515.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0516.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0517.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0518.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0519.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0520.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0521.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0522.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0523.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0524.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0525.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0526.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0527.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0528.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0529.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0530.NEF",
    "D:/images/MFSIG/AA_FLOCK/pappe_70bilder_10x_0.0045dist/src/DSC_0531.NEF"]


    print(files)
    import time

    start1 = time.perf_counter_ns()

    for p in files:
        with open(p, "rb") as f:
            raw_data = f.read()
            
    print((time.perf_counter_ns()-start1)*10**-9)

    start2 = time.perf_counter_ns()
    pool = mp.Pool(mp.cpu_count()*2)
    results = pool.imap(load_raw_from_bytesio, files)
    for f, r in zip(files, results):
        print(f, r[0])
    
    print((time.perf_counter_ns()-start2)*10**-9)


if __name__ == "__main__":
    main()