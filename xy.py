from PIL import Image, ImageDraw, ImageChops
import random 
import numpy as np
import streamlit as st

# идеи
# сделать круг, треугольник, произвольную от пользователя
def interpolate (start, end, n):
    return np.linspace(start, end, n).astype(int)

def get_rd_matrix(x_points, y_points,  random_dev=10, axis_type = 'x'):
    if axis_type == 'y':
        matrix = np.meshgrid(y_points, x_points)
    else:
        matrix = np.meshgrid(x_points, y_points)
    matrix = np.array(matrix)
    
    if random_dev != 0:
        rand_noise = np.random.randint(-random_dev, random_dev, matrix.size).reshape(matrix.shape)
        matrix_rd = matrix + rand_noise
    else: 
        matrix_rd = matrix
    
    return matrix_rd

def interpolate_matrix (matrix_rd, n_split, axis_type = 'x'):
    if axis_type == 'y':
        matrix_rd = np.array([matrix_rd[0].T, matrix_rd[1].T])[::-1]
    else:
        pass
    matrix_rd_interpolate = list()
    for axis in matrix_rd:
        axis_interpolate = list()
        for layer in axis:
            layer_interpolate = np.array([])
            for n, i in enumerate(layer[:-1]):
                a = interpolate(layer[n], layer[n+1], n_split)
                layer_interpolate = np.append(layer_interpolate, a)
            axis_interpolate.append(layer_interpolate)
        matrix_rd_interpolate.append(axis_interpolate) 

    matrix_rd_interpolate = np.array(matrix_rd_interpolate).astype(int)

    if axis_type == 'y':
        matrix_rd_interpolate = matrix_rd_interpolate[::-1]
    
    a = matrix_rd_interpolate.shape
    points_matrix = list()
    for i in range(a[1]):
        layer_points = list()
        for j in range(a[2]):
            point = matrix_rd_interpolate[:,i][:,j]
            layer_points.append(point)
        points_matrix.append(layer_points)

    return points_matrix

def random_color(min_color = 0, max_color = 255):
    return (random.randint(min_color,max_color), random.randint(min_color,max_color), random.randint(min_color,max_color))

def grid_art(n_columns=5, n_rows=5, n_split=10, random_dev=10, image_size = (1080, 1080), main_shape = 'square', min_color = 30, max_color = 60, dev_ratio = 0, width_line = 0, cut_coef = 0.15, matrix_user = ('x', 'x', 'x', 'x'), rgb_background_user=(190, 170, 150), random_seed = 42):
    print('xy!')
    np.random.seed(random_seed)
    random.seed(random_seed)
    image_size_px , image_size_py = image_size
    # parametrs for figure
    # shape of figure
    # main_shape = 'square' #1
    # n_columns = 5 + 1
    # n_rows = 7 + 1
    # n interpolate
    # n_split = 40
    # limits for random_color 
    # min_color = 30
    # max_color = 60
    # # space between boxs
    # dev_ratio = 0
    # # width of lines
    # width_line = 0
    # # shape proportion
    # cut_coef = 0.15
    # deviation of lines
    # random_dev = 100
    
    # i, j = col, row 
    # 1, 2 = нечет, четный
    i2j2, i2j1, i1j1, i1j2 = matrix_user
    r_user, g_user, b_user = rgb_background_user
    r = np.random.normal(r_user, 10, (image_size_px, image_size_py))
    g = np.random.normal(g_user, 10, (image_size_px, image_size_py))
    b = np.random.normal(b_user, 10, (image_size_px, image_size_py))
    rgb = np.array([r, g, b], dtype='uint8')
    noise = rgb.T.reshape(image_size_py, image_size_px, 3)
    image  = Image.fromarray(noise)
    draw = ImageDraw.Draw(image)

    # без шума однотонный фон
    # image_bg_color = (204, 204, 204)
    # image = Image.new("RGB", size = (image_size_px, image_size_py), color = image_bg_color)
    # draw = ImageDraw.Draw(image)
    

    # координаты матрицы для блоков
    x_min = int(cut_coef*image_size_px)
    x_max = int((1-cut_coef)*image_size_px)
    x_points = np.linspace(x_min, x_max, n_columns).astype(int)
    
    if main_shape == 'square':
        # квадрат
        y_min = image_size_py//2 - (x_max - x_min)//2
        y_max = image_size_py//2 + (x_max - x_min)//2
    else:
        # по форме изображения
        y_min = int(cut_coef*image_size_py)
        y_max = int((1-cut_coef)*image_size_py)

    y_points = np.linspace(y_min, y_max, n_rows).astype(int)
    # формирование матрицы
    matrix_rd = get_rd_matrix(x_points, y_points, random_dev)

    all_draw_points = list()
    for i in range(n_columns-1):
        for j in range(n_rows-1):
        
            sample = np.array([[matrix_rd[0][j][i:i+2], matrix_rd[0][j+1][i:i+2]], 
                           [matrix_rd[1][j][i:i+2], matrix_rd[1][j+1][i:i+2]]])

            dev_plus = dev_ratio
            dev_minus = - dev_ratio
            sample = sample + [[[dev_plus, dev_minus], [dev_plus, dev_minus]], 
                    [[dev_plus, dev_plus], [dev_minus, dev_minus]]]
            sample = sample.astype(int)

            if i%2 == 0:
                if j%2 == 0:
                    draw_points = interpolate_matrix(sample, n_split, axis_type = i2j2)
                else:
                    draw_points = interpolate_matrix(sample, n_split, axis_type = i2j1)
            else:
                if j%2 != 0:
                    draw_points = interpolate_matrix(sample, n_split, axis_type = i1j1)
                else:
                    draw_points = interpolate_matrix(sample, n_split, axis_type = i1j2)
                
            all_draw_points.append(draw_points)

    for num, draw_points in enumerate(all_draw_points):
        cols = np.array(draw_points).shape[1]
        start_color = random_color(min_color, max_color)
        end_color = random_color(min_color, max_color)
        
        for col in range(np.array(draw_points).shape[1]):
            a = [tuple (i) for i in list(np.array(draw_points)[:,col])]
            # a = a-10
            color_line = (int(start_color[0]*col/cols) + int(end_color[0]*(cols-col)/cols) ,
                        int(start_color[1]*col/cols) + int(end_color[1]*(cols-col)/cols),
                        int(start_color[2]*col/cols) + int(end_color[2]*(cols-col)/cols))
            draw.line(a, color_line, width = width_line)
    image.save('xy.png') 
    return image
     
def user_input_features():
    username = st.text_input('text your wish', 'be happppyyyy!', 150)
    random_seed = 42
    random.seed(username)
    image_size_px = st.sidebar.slider('px', 500, 2000, 1080)
    image_size_py = st.sidebar.slider('py', 500, 2000, 1080)
    image_size = (image_size_px, image_size_py)

    n_columns_rand, n_rows_rand, n_split_rand = random.randint(0,50), random.randint(0,50), random.randint(0,50)
    n_columns = st.sidebar.slider('n_columns', 0, 50, n_columns_rand)
    n_rows = st.sidebar.slider('n_rows', 0, 50, n_rows_rand)
    n_split = st.sidebar.slider('n_split', 0, 50, n_split_rand)
    random_dev = st.sidebar.slider('random_dev', 0, 200, random.randint(0,200))
    min_color = st.sidebar.slider('min_color', 0, 140, random.randint(0,140))
    max_color = st.sidebar.slider('max_color', 140, 255, random.randint(140,255))
    # space between boxs
    dev_ratio = st.sidebar.slider('dev_ratio', 0, 50, random.randint(0,50))
    # width of lines
    width_line = st.sidebar.slider('width', 0, 60, random.randint(0,60))
    # shape proportion
    cut_coef = 0.15
    # ticker_name = st.sidebar.text_input("Ticker", "NEE")
    main_shape = st.sidebar.selectbox('main_shape', ('square','proportion'))
    i2j2 = st.sidebar.selectbox('i2j2', ('x','y'))
    i2j1 = st.sidebar.selectbox('i2j1', ('x','y'))
    i1j1 = st.sidebar.selectbox('i1j1', ('x','y'))
    i1j2 = st.sidebar.selectbox('i1j2', ('x','y'))
    matrix_user = (i2j2, i2j1, i1j1, i1j2)
    r_user = st.sidebar.slider('r_back', 0, 255, random.randint(0,255))
    g_user = st.sidebar.slider('g_back', 0, 255, random.randint(0,255))
    b_user = st.sidebar.slider('b_back', 0, 255, random.randint(0,255))
    rgb_background_user = (r_user, g_user, b_user)
    image = grid_art(n_columns, n_rows, n_split, random_dev, image_size, main_shape, min_color, max_color, dev_ratio, width_line, cut_coef, matrix_user, rgb_background_user, random_seed)
    
    return image



if __name__ == '__main__':
    st.write("""# Take your wish reminder for 2025!""")
    st.write('contact https://t.me/gregoryish')
    fig = user_input_features()
    st.image(fig, caption='Your Wish Reminder')
    
    
    
