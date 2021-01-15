class Settings():
    """存储所有设置的类"""

    def __init__(self):
        """初始化游戏的设置"""
        # 屏幕设置
        self.screen_width = 1200
        self.screen_height = 800

        # 子弹设置
        self.bullet_speed_factor = 5
        self.bullet_width = 3
        self.bullet_height = 15

        # 外星人设置
        self.alien_speed_factor = 10
        self.fleet_drop_speed = 20
        # fleet_direction为1表示向右移，为-1表示向左移
        self.fleet_direction = 1

        # 飞船设置
        self.ship_speed_factor = 5
        self.ship_limit = 0
