import numpy as np
import cv2
import copy
from create_video import create_video
from progress.bar import Bar


vc = cv2.VideoCapture('test4.avi')
bg_sub = cv2.bgsegm.createBackgroundSubtractorMOG()
length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

bar = Bar('Reading Frames', max=length)

iteration_first = 1
for i in range(0, length):
    ret, fr = vc.read()
    # If first frame
    if iteration_first == 1:
        firstfr = copy.deepcopy(fr)
        h, w = fr.shape[:2]
        mhi_image = np.zeros((h, w), np.uint8)
        iteration_first = 0
    else:
        filt = bg_sub.apply(fr)  # remove the background
        cv2.imwrite('frame.jpg', fr)
        cv2.imwrite('diff-frame.jpg', filt)
        
        thresh = 2
        maxVal = 2
        ret, thresh1 = cv2.threshold(filt, thresh, maxVal, cv2.THRESH_BINARY)
        
        # add to the accumulated image
        mhi_image = cv2.add(mhi_image, thresh1)
        cv2.imwrite('mask.jpg', mhi_image)
        color_image_video = cv2.applyColorMap(mhi_image, cv2.COLORMAP_HOT)
        
        vfr = cv2.addWeighted(fr, 0.7, color_image_video, 0.7, 0)
        name = "frames/frame%d.jpg" % i
        cv2.imwrite(name, vfr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        bar.next()
    
bar.finish()
create_video('frames/', './output.avi')
color_image = cv2.applyColorMap(mhi_image, cv2.COLORMAP_OCEAN)
result_overlay = cv2.addWeighted(firstfr, 0.7, color_image, 0.7, 0)
    
# save the final heatmap
cv2.imwrite('diff-overlay.jpg', result_overlay)
# cleanup
vc.release()
cv2.destroyAllWindows()
