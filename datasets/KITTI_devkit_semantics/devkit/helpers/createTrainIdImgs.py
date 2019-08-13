import os, cv2
from devkit.helpers.labels import id2label

#! 将KITTI数据集的label图像转为datasets/KITTI_devkit_semantics/devkit/helpers/labels.py里面定义的trainId

def is_label_kitti(filename):
    return filename.endswith("_labelIds.png")


def main():
    KittiPath = '/home/gongyiqun/images/KITTI/data_semantics/training/gtFine'

    train_dir = os.path.join(KittiPath, "train")
    val_dir = os.path.join(KittiPath, "val")

    train_file_list = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f)) and is_label_kitti(f)]
    val_file_list = [f for f in os.listdir(val_dir) if os.path.isfile(os.path.join(val_dir, f)) and is_label_kitti(f) ]
    train_file_list.sort()
    val_file_list.sort()


    for f in train_file_list:
        semantic_img = cv2.imread( os.path.join(train_dir, f) )
        # cv2.imshow("src", semantic_img)
        print("proceeding " + f)
        output = f.replace("_labelIds.png", "_labelTrainIds.png")

        for i in range(semantic_img.shape[0]):
            for j in range(semantic_img.shape[1]):
                for c in range(semantic_img.shape[2]):
                    pix = semantic_img[i,j,c]
                    new_pix = id2label[pix].trainId
                    if new_pix < 19 or new_pix == 255:
                        semantic_img[i, j, c] = new_pix
                    else:
                        print("*************error**************")

        # cv2.imshow("rgb", semantic_img)
        semantic_img = cv2.cvtColor(semantic_img, cv2.COLOR_RGB2GRAY)
        cv2.waitKey(5)
        cv2.imwrite(os.path.join(train_dir, output), semantic_img)

    for f in val_file_list:
        semantic_img = cv2.imread( os.path.join(val_dir, f) )
        # cv2.imshow("src", semantic_img)
        print("proceeding " + f)
        output = f.replace("_labelIds.png", "_labelTrainIds.png")

        for i in range(semantic_img.shape[0]):
            for j in range(semantic_img.shape[1]):
                for c in range(semantic_img.shape[2]):
                    pix = semantic_img[i,j,c]
                    new_pix = id2label[pix].trainId
                    if new_pix < 19 or new_pix == 255:
                        semantic_img[i, j, c] = new_pix
                    else:
                        print("*************error**************")

        # cv2.imshow("rgb", semantic_img)
        semantic_img = cv2.cvtColor(semantic_img, cv2.COLOR_RGB2GRAY)
        cv2.waitKey(5)
        cv2.imwrite(os.path.join(val_dir, output), semantic_img)

# call the main
if __name__ == "__main__":
    main()
