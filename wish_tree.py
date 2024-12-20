from PIL import Image, ImageDraw
from PIL import Image, ImageDraw
import numpy as np
import random
import math 
import streamlit as st

# gradient background
gradient_json = { 
    'intervals':{
        0: {
            'start_color': (255,93,62),
            'end_color': (219,177,135),
            'proportion': 20
            },
        1: {
            'start_color':(219,177,135),
            'end_color':(190,194,197),
            'proportion':15
            },
        2: {
            'start_color': (190,194,197),
            'end_color': (58,121,187),
            'proportion': 35
            },
        3: {
            'start_color': (58,121,187),
            'end_color': (32,69,104),
            'proportion': 30
            }
    }
}

def create_gradient_matrix(rows, cols, start_color, end_color):
    # Create an empty matrix with the specified dimensions
    matrix = np.zeros((rows, cols))
    
    # Calculate the gradient step
    gradient_step = (start_color - end_color) / (rows - 1)
    
    # Fill the matrix with the gradient values
    for i in range(rows):
        value = start_color - i * gradient_step
        matrix[i, :] = value
    
    return matrix



def get_interval_matrix(image_size_px, interval_num, rgb):
    start_color = gradient_json['intervals'][interval_num]['start_color'][rgb]
    end_color = gradient_json['intervals'][interval_num]['end_color'][rgb]
    proportion = gradient_json['intervals'][interval_num]['proportion']

    # Example usage
    rows, cols = round(image_size_px*proportion/100), image_size_px
    print(rows)
    gradient_matrix_interval = create_gradient_matrix(rows, cols, start_color, end_color)
    return gradient_matrix_interval




def new_stick(prev_stick, coef, side = 'random', rarity = 0.25,  type_node = 'half',user_seed=42):
    """
    prev_stick = coordinates of previous stick ((x1, y1), (x2, y2)) 
    coef = cut factor relate to the length of previous stick
    side = in which side the tree stick will grow ('random', 'left', 'right')
    rarity = the probability that the tree stick will grow
    type_node = 'half', 'end' this parametr defines the starting point of new tree stick relative to the previous one
    """
    # random.seed(user_seed)
    # prev_stick = ((x1,y1), (x2,y2))

    # p1 
    if type_node == 'half':
        p1 = (int((prev_stick[0][0] + prev_stick[1][0])/2), int((prev_stick[0][1] + prev_stick[1][1])/2))
    elif type_node == 'end':
        p1 = prev_stick[1]
    

    prev_stick_length = ((prev_stick[0][1] - prev_stick[1][1])**2 + (prev_stick[0][0] - prev_stick[1][0])**2)**0.5
    new_stick_length = (coef)*prev_stick_length

    pi = math.pi
    min_rad = -pi/6
    max_rad = 7*pi/6

    if side == 'left':
        rad = random.uniform(min_rad, pi/2)
    elif side == 'right':
        rad = random.uniform(pi/2, max_rad)
    elif side == 'random': 
        rad = random.uniform(min_rad, max_rad)

    # proections
    dy = math.sin(rad) * new_stick_length
    dx = math.cos(rad) * new_stick_length

    x2 = prev_stick[1][0] - int(dx)
    y2 = prev_stick[1][1] - int(dy)
    p2 = (x2,y2)
    # random.seed(42)
    if random.random() < (rarity):
        new_stick = (p1,p2)
    else:
        new_stick = None
    return new_stick

def get_lvl_color (level, num_levels, start_color, end_color):
    "gradient - this function define color for each level "
    a = level/(num_levels-1)
    b = 1-a
    lvl_color = (
                int(b * start_color[0] + a * end_color[0]),
                int(b * start_color[1] + a * end_color[1]),
                int(b * start_color[2] + a * end_color[2])
    )
    return lvl_color

# generate_art(levels, cut_coef, image_size_px, width_base, random_seed)

def generate_art(levels=10, cut_coef = 0.3, image_size_px=2048, width_base=10, user_seed=42):
    print("Polotno Art!")
    # np.random.seed(user_seed)
    random.seed(user_seed)
    # features

    # cut_coef
    # start_color
    # end_color = (random.randint(90,150), random.randint(90,150),random.randint(90,150))
    # levels
    # image_size_px

    # cut_coef = cut_coef
    # image_bg_color = (85, 80, 90)
    start_color = (random.randint(0,60), random.randint(0,60), random.randint(0,60))
    end_color = (random.randint(90,150), random.randint(90,150),random.randint(90,150))
    # start_color = (252, 248, 3)
    # end_color = (200, 162, 200)

    #=========================
    # image_size_px = 2160

    # r = np.random.normal(135,10, (image_size_px, image_size_px))
    # g = np.random.normal(140,10, (image_size_px, image_size_px))
    # b = np.random.normal(140,10, (image_size_px, image_size_px))
    # rgb = np.array([r, g, b], dtype='uint8')
    
    # noise = rgb.T.reshape(image_size_px, image_size_px, 3)

    # image = Image.fromarray(noise)
    #===============================
    # image_size_px = 2048*2s
    count_intervals = len(gradient_json['intervals'])

    r = []
    g = []
    b = []

    for interval_i in range(count_intervals):
        print(interval_i)
        
        r_i = get_interval_matrix(image_size_px, interval_i,0)
        g_i = get_interval_matrix(image_size_px, interval_i,1)
        b_i = get_interval_matrix(image_size_px, interval_i,2)
        if len(r)>0 :
            r = np.concatenate((r,  r_i), axis=0)
            g = np.concatenate((g,  g_i), axis=0)
            b = np.concatenate((b,  b_i), axis=0)
        else:
            r = r_i
            g = g_i
            b = b_i

    

    rgb = np.array([r, g, b], dtype='uint8')
    noise = rgb.T.reshape(image_size_px, image_size_px, 3)
    image = Image.fromarray(noise)


    # image = image.convert('RGB')
    # image = Image.new("RGB", size = (image_size_px, image_size_px), color = image_bg_color)

    levels = levels

    tree_sticks = {}

    for lev in range(levels):
        print(lev)
        if lev == 0:
            stick = (
                        (int(image_size_px/2),image_size_px), 
                        (int(image_size_px/2)+random.randint(-40,40),
                            image_size_px - int(image_size_px*(1-cut_coef*2))))
            tree_sticks[lev] = [stick]
        else:
            prev_sticks = tree_sticks[lev-1]
            new_sticks = []
            for prev_stick in prev_sticks:


                new_stick1 = new_stick(prev_stick, cut_coef, side = 'left', rarity = ((lev - lev/2)**2)**0.5, user_seed=user_seed)
                new_stick2 = new_stick(prev_stick, cut_coef, side = 'right', rarity = ((lev - lev/2)**2)**0.5, user_seed=user_seed)
                new_stick3 = new_stick(prev_stick, cut_coef, side = 'random', rarity = ((lev - lev/2)**2)**0.5, user_seed=user_seed)
                
                # new_stick4 = new_stick(prev_stick, cut_coef, side = 'left', rarity = 0.8, type_node = 'end')
                # new_stick5 = new_stick(prev_stick, cut_coef, side = 'right', rarity = 0.5, type_node = 'end')
                # new_stick6 = new_stick(prev_stick, cut_coef, side = 'random', rarity = ((lev - lev/2)**2)**0.5 , lvl = lev, levels = levels,  type_node = 'end')
                
                if new_stick1 is not None:
                    new_sticks.append(new_stick1)
                else:
                    pass
                if new_stick2 is not None:
                    new_sticks.append(new_stick2)
                else:
                    pass
                if new_stick3 is not None:
                    new_sticks.append(new_stick3)
                else:
                    pass

            tree_sticks[lev] = new_sticks
    
    image = image.rotate(90)
    draw = ImageDraw.Draw(image)
    
    for i, lev in enumerate(tree_sticks):
        lvl_color = get_lvl_color(lev, levels, start_color, end_color)
        
        for stick in tree_sticks[lev]:
            width_stick = width_base*(levels-i)/levels
            draw.line(stick, fill = lvl_color, width = int(width_stick) )

    image.save('polotno.png')
    return image



def user_input_features():
    user_seed = st.text_input('print your wish for 2025 here', 'beeee happppyyy!', 150)
    random_seed = 42
    random.seed(user_seed)
    image_size_px = st.sidebar.slider('size', 500, 4096, 2048)
    # image_size_py = st.sidebar.slider('py', 500, 2000, 1080)
    
    cut_coef_rnd = random.randint(0,50)
    cut_coef = st.sidebar.slider('regulate shape', 0, 50, cut_coef_rnd)/100
    levels_rnd = random.randint(5,12)
    levels = st.sidebar.slider('levels of tree', 5, 12, levels_rnd)
    width_base_rnd = random.randint(levels-5,levels+15)
    width_base = st.sidebar.slider('width of tree', 1, 50, width_base_rnd)
    # rarity_rnd = random.randint(20,60)/100
    # rarity = st.sidebar.slider('probability of new stick', 20, 60, rarity_rnd)/100
    image = generate_art(levels, cut_coef, image_size_px, width_base, user_seed)
    
    return image 

if __name__ == '__main__':
    st.write("""# Take your wish tree for 2025!""")
    st.write('contact https://t.me/gregoryish')
    fig = user_input_features()
    st.image(fig, caption='Your Wish Tree')

