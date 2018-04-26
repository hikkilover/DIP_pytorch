import os
from urllib.request import urlretrieve

if __name__ == '__main__':
    if 'denoising_BM3D' not in os.listdir():
        os.makedirs('denoising_BM3D')

    baseURL = 'http://www.cs.tut.fi/~foi/GCF-BM3D/images/'
    imageList = ['image_Lena512rgb', 'image_House256rgb', 'image_Peppers512rgb',
                 'image_Baboon512rgb', 'image_F16_512rgb', 'kodim01',
                 'kodim02', 'kodim03', 'kodim12']
    NSList = ['5', '10', '15', '20', '25', '30', '35', '40',
              '50', '60', '70', '75', '80', '90', '100']

    for image_name in imageList:
        base_path = 'denoising_BM3D' + '/' + image_name.replace('image_', '') + '/'
        if image_name not in os.listdir('denoising_BM3D'):
            os.makedirs(base_path)
        for ns in NSList:
            server_file_name = '%s_noi_s%s.png' % (image_name, ns)
            local_file_name = '%s_noi_s%s.png' % (image_name.replace('image_', ''), ns)
            target_url = baseURL + server_file_name
            urlretrieve(url=target_url, filename=base_path + local_file_name)
            print(local_file_name + '  Download Success!')