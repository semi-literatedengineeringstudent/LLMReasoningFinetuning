from pynput.mouse import Controller, Button
import time 
mouse = Controller()
# reference： https://www.youtube.com/watch?v=78rSqtkw3Gk
while True:
    mouse.click(Button.left, 1)
    print('click')
    time.sleep(5)