from scipy.interpolate import Akima1DInterpolator
import numpy as np
import pandas as pd

def interpolate_abnormal_keypoints(keypoints_data):
    interpolated_data = []
    
    for video_id, group in keypoints_data.groupby('video_id'):
        sorted_group = group.sort_values(by='frame_id')
        keypoints = np.array(sorted_group['keypoints'].tolist())

        for j in range(0, keypoints.shape[1], 3):  # 每3个数处理一次 (x, y, 置信度)
            x = keypoints[:, j]
            y = keypoints[:, j+1]
            confidence = keypoints[:, j+2]
            
            valid_indices = np.where(confidence > 0)[0]  # 定义有效点为置信度大于0的点
            
            if len(valid_indices) > 1:
                try:
                    akima_x = Akima1DInterpolator(valid_indices, x[valid_indices])
                    akima_y = Akima1DInterpolator(valid_indices, y[valid_indices])

                    all_indices = np.arange(len(x))
                    x_interpolated = akima_x(all_indices)
                    y_interpolated = akima_y(all_indices)

                    # 检查插值结果是否包含NaN
                    if not (np.isnan(x_interpolated).any() or np.isnan(y_interpolated).any()):
                        keypoints[:, j] = x_interpolated
                        keypoints[:, j+1] = y_interpolated
                    else:
                        print("NaN detected after interpolation, skipping update for these keypoints.")
                        # skip 18
                except ValueError as e:
                    # 捕获可能的插值错误
                    print(f"Interpolation error: {e}")
                    pass  # 可以选择跳过或记录错误
            else:
                # 不足两个有效点，跳过或其他处理
                pass

        for i, row in enumerate(sorted_group.to_dict('records')):
            row['keypoints'] = keypoints[i].tolist()
            interpolated_data.append(row)

    return pd.DataFrame(interpolated_data)
