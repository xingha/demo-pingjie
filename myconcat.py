import cv2
import numpy as np

def img_concat(img_left0, img_right0):
    """
    全景图拼接过程：
    1.提取sift算子特征向量、特征关键点；
    2.对两张图片的关键点做匹配，使用knnmatch；
    3.根据匹配结果，对关键点进行筛选；
    4.然后根据重排结果计算H变换矩阵；
    5.根据H矩阵计算右图的透视图；
    6.将左图和透视图进行拼接；
    """
    # 0. 图片预处理
    img_left = cv2.cvtColor(img_left0, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right0, cv2.COLOR_BGR2GRAY)

    # 1.提取sift算子关键点、特征向量
    siftor = cv2.SIFT_create()
    keypoints_left, features_left = siftor.detectAndCompute(img_left, None)
    keypoints_right, features_right = siftor.detectAndCompute(img_right, None)

    ## 1.1 画出关键点
    img_left_keypoints = cv2.drawKeypoints(img_left, keypoints_left, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    img_right_keypoints = cv2.drawKeypoints(img_right, keypoints_right, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    ## 1.2 画出特征向量
    left_concat = np.hstack((img_left0, img_left_keypoints))
    right_concat = np.hstack((img_right0, img_right_keypoints))
    cv2.imshow('left_concat', left_concat)
    cv2.imshow('right_concat', right_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # 2.对两张图片的关键点做匹配，使用knnmatch，K=2取两个点,以右图作为模板
    matchor = cv2.BFMatcher()
    matches = matchor.knnMatch(features_right, features_left, k=2)

    ## 2.2 对匹配点按照两个关键点的差异度进行排序
    matches = sorted(matches, key=lambda x: x[0].distance/x[1].distance)

    # 3.根据匹配结果，对关键点进行筛选
    good_matches = []
    match_thresh = 0.6
    for m, n in matches:
        if m.distance < match_thresh * n.distance:
            good_matches.append(m)
    
    ## 3.1 画出匹配点
    img_match = cv2.drawMatches(img_right0, keypoints_right, img_left0, keypoints_left, good_matches, None, None, None, None, flags=2)
    cv2.imshow('img_match', img_match)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 4.然后根据重排结果计算H变换矩阵
    homography_thresh = 4  # 大于4
    if len(good_matches) > homography_thresh:
        src_ptsL = np.float32([keypoints_left[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_ptsR = np.float32([keypoints_right[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        ransacReprojThreshold = 4
        H, mask = cv2.findHomography(dst_ptsR, src_ptsL, cv2.RANSAC, ransacReprojThreshold)

        # 5.根据H矩阵计算右图的透视图
        transform = cv2.warpPerspective(img_right0, H, (img_left.shape[1] + img_right.shape[1], img_left.shape[0]))
        cv2.imshow('transformed', transform)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

        # 6.将左图和透视图进行拼接
        transform[0:img_left0.shape[0], 0:img_left0.shape[1]] = img_left0
        cv2.imshow('complete', transform)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 7.保存拼接图
        cv2.imwrite('img/myconcat.jpg', img_concat)
    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches), homography_thresh))


if __name__ == '__main__':
    img_left = cv2.imread('img/src1.jpg')
    img_right = cv2.imread('img/det.jpg')
    img_left = cv2.resize(img_left, (0, 0), fx=0.8, fy=0.6)
    img_right = cv2.resize(img_right, (img_left.shape[1], img_left.shape[0]))
    img_concat(img_left, img_right)

