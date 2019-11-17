import sys
import os
import pickle
import pandas as pd
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


model = load_model('mnist_digits_multi_bw.joblib')

pygame.init()

image_x = 28
image_y = 28

window_scale = 10

screen_width = image_x * window_scale
screen_height = image_y * window_scale
size = width, height = screen_width, screen_height

cursor_radius = 2
last_pos_x = -1
last_pos_y = -1
prediction = 0

transparent = (0, 0, 0, 0)
black = (0, 0, 0)
white = (255, 255, 255)
yellow = (255, 255, 0)

font = pygame.font.Font('freesansbold.ttf', 32)

screen = pygame.display.set_mode(size)

canvas = pygame.Surface((image_x, image_y))
canvas.fill(black)


def draw(x, y):
    canvas.set_at((x + 1, y + 1), white)
    canvas.set_at((x + 1, y), white)
    canvas.set_at((x, y + 1), white)
    canvas.set_at((x, y), white)


def predict_number():

    image = pygame.surfarray.array2d(canvas).transpose().flatten()
    for i in range(0, len(image)):
        image[i] = image[i]//65793

    return model.predict(image.reshape(1, -1))[0]


while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    mousex, mousey = pygame.mouse.get_pos()
    mouse_image_x = mousex // window_scale
    mouse_image_y = mousey // window_scale

    left_bt, middle_bt, right_bt = pygame.mouse.get_pressed()

    if (left_bt or right_bt) and (mouse_image_x != last_pos_x or mouse_image_y != last_pos_y):
        if left_bt:
            draw(mouse_image_x, mouse_image_y)
        elif right_bt:
            pygame.draw.circle(canvas, black, (mouse_image_x, mouse_image_y), 2)

        last_pos_x = mouse_image_x
        last_pos_y = mouse_image_y

        prediction = predict_number()
    elif middle_bt:
        canvas.fill(black)

    predict = font.render(str(prediction), True, yellow, transparent)
    predict_rect = predict.get_rect()
    predict_rect.bottomright= (screen_width, screen_height)

    screen.blit(pygame.transform.scale(canvas, (280, 280)), (0, 0))
    screen.blit(predict, predict_rect)
    pygame.display.flip()
