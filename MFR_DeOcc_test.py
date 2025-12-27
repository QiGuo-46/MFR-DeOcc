import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import numpy as np
sys.path.append('..')
from utils.dataset import dataset as dataset
from model.model_AC import model as M
import cv2
import matplotlib.pyplot as plt

def saveImg(img, path):
    img[img > 1] = 1
    img[img < 0] = 0
    op = np.zeros([img.shape[2], img.shape[3], 3])
    img = img[0].cpu().permute(1,2,0) * 255
    op[:,:,0] = img[:,:,2]
    op[:,:,1] = img[:,:,1]
    op[:,:,2] = img[:,:,0]
    cv2.imwrite(path, op)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    import torch
    from torch import nn
    from PIL import Image
    from utils import psnr_batch, ssim_batch
    import cv2

    device = 'cuda'

    run_dir = './log/2024-02-01_3/'     #./log/2024-02-01_3/

    testFolder = dataset('./test1.txt')
    testLoader = torch.utils.data.DataLoader(testFolder, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    model = M(netParams={'Ts': 1, 'tSample': 40}, inChannels=33+11, norm='BN')
    model = torch.nn.DataParallel(model)
    model.cuda()


    print('==> loading existing model:', os.path.join(run_dir, 'checkpoint.pth.tar'))  #checkpoint.pth.tar
    model_info = torch.load(os.path.join(run_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(model_info['state_dict'],False)

    savePath = os.path.join(run_dir, 'output')
    savePath1 = os.path.join(run_dir, 'event_vox')
    os.makedirs(savePath, exist_ok=True)

    with torch.no_grad():
        model.eval()
        psnr = 0
        ssim = 0
        count = 0

        psnr_indoor = 0
        ssim_indoor = 0
        count_indoor = 0

        psnr_outdoor = 0
        ssim_outdoor = 0
        count_outdoor = 0
        for i, (event_vox, img, gt_img, mask) in enumerate(testLoader):

            ep_data=event_vox.squeeze(0)
            print(ep_data.shape)  #t=40


            for j in range(40):
                ep_data1 = ep_data[:, :, :, j].numpy()
                print(ep_data1.shape)

                ergb_image = np.stack((ep_data1[0]*256, np.zeros_like(ep_data1[0]), ep_data1[1]*256 ), axis=-1)

                cv2.imwrite("./%devent%d.png"%(i,j), ergb_image)  #t=33/3=11

            print(img.shape)

            for j in range(11):

                numpy_array1 = np.zeros([img.shape[2], img.shape[3], 3])
                numpy_array = img.squeeze().permute(1, 2, 0).numpy() * 255
                print(numpy_array.shape)
                numpy_array1[:, :, 0] = numpy_array[:, :, 3*j+2]
                numpy_array1[:, :, 1] = numpy_array[:, :, 3*j+1]
                numpy_array1[:, :, 2] = numpy_array[:, :, 3*j]
                cv2.imwrite("./%dimg%d.png"%(i,j), numpy_array1[:,:,0:3])
                print(numpy_array1.shape)  #t=11

                print(j)
                img0=cv2.convertScaleAbs(numpy_array[:, :, 3*j:3*j+3])
            # 转换为灰度图像
                gray_image = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                print(gray_image)

            # 二值化
                _, binary_image = cv2.threshold(gray_image, 0.39*255, 255, cv2.THRESH_BINARY)
                print(binary_image)
                image_mask0 = binary_image
                image_mask0= cv2.convertScaleAbs(binary_image)
                image_mask0=255-image_mask0
                cv2.imwrite('mask%d.png' % j, image_mask0)
            # 显示图像

            print(gt_img.shape)
            gt_img0= gt_img[0].squeeze().permute(1, 2, 0).numpy() * 255
            gt_img1 = gt_img[0].squeeze().permute(1, 2, 0).numpy() * 255
            print(gt_img0.shape)
            gt_img1[:, :, 0] = gt_img0[:, :, 2]
            gt_img1[:, :, 1] = gt_img0[:, :, 1]
            gt_img1[:, :, 2] = gt_img0[:, :, 0]
            cv2.imwrite('gt_img%d.png' % i, gt_img1)

            event_vox = event_vox.cuda()
            img = img.cuda().float()
            mask = mask.cuda().float()
            print(mask)
            gt_img = gt_img.cuda().float()

            mask = torch.index_select(mask, 1, torch.tensor([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]).cuda())
            print(mask.shape)
            output = model(event_vox, torch.cat([img, mask], dim=1))

            #p = psnr_batch(output, gt_img)
            #s = ssim_batch(output, gt_img)


            saveImg(output, os.path.join(savePath, '%d_output.png'%i))
            saveImg(gt_img, os.path.join(savePath, '%d_gt.png'%i))
            #saveImg(image_mask0, os.path.join(savePath, '%d_mask0.png' % i))
            cv2.imwrite('%d_mask0.png' % i,image_mask0)

"""
            print(i, p, s)
            psnr += p
            ssim += s
            count += 1

            if i <= 16:
                psnr_indoor += p
                ssim_indoor += s
                count_indoor += 1
            else:
                psnr_outdoor += p
                ssim_outdoor += s
                count_outdoor += 1

        psnr /= count
        ssim /= count
        print(psnr, ssim, psnr_indoor/count_indoor, ssim_indoor/count_indoor, psnr_outdoor/count_outdoor, ssim_outdoor/count_outdoor)
        
"""

#CUDA_VISIBLE_DEVICES=0,1 python VEFNet_train.py    i-1.gpushare.com
#
"""
conda activate guoqi
cd /root/Event_Enhanced_DeOcc-main/
CUDA_VISIBLE_DEVICES=5 python VEFNet_test3.py
"""