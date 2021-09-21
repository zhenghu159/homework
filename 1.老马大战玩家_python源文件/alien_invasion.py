import sys
import pygame
import time
from settings import Settings
from game_stats import GameStats
from button import Button
from ship import Ship
from alien import Alien
from pygame.sprite import Group
import game_functions as gf

def run_game():
    # 初始化游戏并创建一个屏幕对象
    pygame.init() # 初始化背景设置
    ai_settings = Settings()
    screen = pygame.display.set_mode((ai_settings.screen_width, ai_settings.screen_height)) # 创建一个宽1200像素、高800像素的游戏窗口
    pygame.display.set_caption("玩家大战老马") # 设置窗口名称
    # 创建Play按钮
    play_button = Button(ai_settings, screen, "Play")
    # 创建一艘飞船、一个子弹编组和一个外星人编组
    ship = Ship(ai_settings, screen)
    bullets = Group()
    aliens = Group()
    # 创建外星人群
    gf.create_fleet(ai_settings, screen,ship, aliens)
    # 创建一个用于存储游戏统计信息的实例
    stats = GameStats(ai_settings)
    # 游戏音乐
    pygame.mixer.init()  # 初始化
    pygame.mixer.music.load('./sounds/bgm.mp3')  # 加载音乐文件
    pygame.mixer.music.play()  # 开始播放音乐流

    # 开始游戏的主循环
    while True:
        gf.check_events(ai_settings, screen, stats, play_button, ship, aliens, bullets)
        if stats.game_active:
            ship.update()
            gf.update_bullets(ai_settings, screen, ship, aliens, bullets)
            gf.update_aliens(ai_settings, stats, screen, ship, aliens, bullets)
        gf.update_screen(ai_settings, screen, stats, ship, aliens, bullets, play_button)

run_game()