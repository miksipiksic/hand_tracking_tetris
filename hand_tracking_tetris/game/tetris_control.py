
import pygame
import random
import sys
import json
import cv2
import numpy as np
import sys

from collections import deque


import sklearn
import joblib
import cv2

import mediapipe as mp
import math

import pyautogui
import os
import time
from datetime import datetime

# ---- CONFIG ----
CELL_SIZE = 30
COLS = 10
ROWS = 20
WIDTH = CELL_SIZE * COLS
HEIGHT = CELL_SIZE * ROWS
FPS = 60

SCORES = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}

PANEL_WIDTH = 600
PANEL_BG = (40, 40, 80)  # soft retro background
TEXT_COLOR = (255, 255, 0)
SUBTEXT_COLOR = (200, 200, 200)

SCREEN_WIDTH = WIDTH + PANEL_WIDTH
SCREEN_HEIGHT = HEIGHT



BLACK = (0, 0, 0)
GRAY = (40, 40, 40)
WHITE = (255, 255, 255)
COLORS = [
    (0, 240, 240),   # I
    (0, 0, 240),     # J
    (240, 160, 0),   # L
    (240, 240, 0),   # O
    (0, 240, 0),     # S
    (160, 0, 240),   # T
    (240, 0, 0),     # Z
]

SHAPES = {
    'I': [[0,0,0,0],
          [1,1,1,1],
          [0,0,0,0],
          [0,0,0,0]],
    'J': [[1,0,0],
          [1,1,1],
          [0,0,0]],
    'L': [[0,0,1],
          [1,1,1],
          [0,0,0]],
    'O': [[1,1],
          [1,1]],
    'S': [[0,1,1],
          [1,1,0],
          [0,0,0]],
    'T': [[0,1,0],
          [1,1,1],
          [0,0,0]],
    'Z': [[1,1,0],
          [0,1,1],
          [0,0,0]],
}

SHAPE_KEYS = list(SHAPES.keys())

RETRO_FONT_TITLE = None
RETRO_FONT_SMALL = None

HIGHSCORES_FILE = "highscores.json"

# ---- HIGH SCORES ----
def load_highscores():
    if os.path.exists(HIGHSCORES_FILE):
        with open(HIGHSCORES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_highscores(scores):
    with open(HIGHSCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

def get_player_name(screen):
    # Load a retro font
    try:
        retro_font_title = pygame.font.Font("PressStart2P.ttf", 24)
        retro_font_input = pygame.font.Font("PressStart2P.ttf", 24)
    except:
        # fallback if ttf not found
        retro_font_title = pygame.font.SysFont("Courier New", 24, bold=True)
        retro_font_input = pygame.font.SysFont("Courier New", 24, bold=True)

    name = ""
    entering = True
    clock = pygame.time.Clock()
    WIDTH, HEIGHT = screen.get_size()

    box_width, box_height = 420, 160
    box_x = (WIDTH - box_width) // 2
    box_y = (HEIGHT - box_height) // 2
    input_box = pygame.Rect(box_x + 20, box_y + 100, box_width - 40, 40)

    while entering:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    entering = False
                elif event.key == pygame.K_BACKSPACE:
                    name = name[:-1]
                else:
                    if len(name) < 12 and event.unicode.isprintable():
                        name += event.unicode

        # Draw shadow
        shadow_rect = pygame.Rect(box_x + 5, box_y + 5, box_width, box_height)
        pygame.draw.rect(screen, (20, 20, 20), shadow_rect, border_radius=10)

        # Draw popup box
        pygame.draw.rect(screen, (30, 30, 60), (box_x, box_y, box_width, box_height), border_radius=10)
        pygame.draw.rect(screen, (200, 200, 255), (box_x, box_y, box_width, box_height), 3, border_radius=10)

        # Prompt text
        txt_label = retro_font_title.render("ENTER YOUR NAME", True, (255, 255, 0))
        label_rect = txt_label.get_rect(center=(box_x + box_width // 2, box_y + 40))
        screen.blit(txt_label, label_rect)

        # Draw input box
        pygame.draw.rect(screen, (0, 0, 0), input_box)
        pygame.draw.rect(screen, (255, 255, 0), input_box, 2)

        # Render typed text and center it
        txt_input = retro_font_input.render(name, True, (0, 255, 0))
        txt_rect = txt_input.get_rect(center=input_box.center)
        screen.blit(txt_input, txt_rect)

        pygame.display.flip()
        clock.tick(30)

    return name if name else "Anon"





def update_highscores(score, screen):
    scores = load_highscores()
    scores = sorted(scores, key=lambda s: s["score"], reverse=True)
    if len(scores) < 5 or score > scores[-1]["score"]:
        name = get_player_name(screen)
        scores.append({"name": name, "score": score})
        scores = sorted(scores, key=lambda s: s["score"], reverse=True)[:5]
        save_highscores(scores)
    return scores

# ---- UTILITIES ----
def rotate(shape):
    return [list(row) for row in zip(*shape[::-1])]

def create_empty_board():
    return [[-1 for _ in range(COLS)] for _ in range(ROWS)]

# ---- PIECE CLASS ----
class Piece:
    def __init__(self, kind=None):
        self.kind = kind if kind else random.choice(SHAPE_KEYS)
        self.shape = [row[:] for row in SHAPES[self.kind]]
        self.color = COLORS[SHAPE_KEYS.index(self.kind)]
        self.x = (COLS // 2) - (len(self.shape[0]) // 2)
        self.y = 0

    def rotate(self):
        self.shape = rotate(self.shape)

    def width(self):
        return len(self.shape[0])

    def height(self):
        return len(self.shape)

# ---- TETRIS GAME ----
class Tetris:
    def __init__(self):
        self.board = create_empty_board()
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.game_over = False
        self.paused = False

        self.current = Piece()
        self.next = Piece()
        self.drop_timer = 0.0
        self.drop_interval = self.compute_drop_interval()
        self.lock_delay = 1.0  # 1 second lock delay
        self.lock_timer = 0.0
        self.landed = False

    def compute_drop_interval(self):
        base = 0.9 ** (self.level - 1)
        return max(0.05, 0.8 * base)

    def can_place(self, shape, x, y):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    bx = x + c
                    by = y + r
                    if bx < 0 or bx >= COLS or by < 0 or by >= ROWS:
                        return False
                    if self.board[by][bx] != -1:
                        return False
        return True

    def place_piece(self):
        for r, row in enumerate(self.current.shape):
            for c, cell in enumerate(row):
                if cell:
                    bx = self.current.x + c
                    by = self.current.y + r
                    if 0 <= by < ROWS and 0 <= bx < COLS:
                        self.board[by][bx] = SHAPE_KEYS.index(self.current.kind)

    def remove_full_lines(self):
        new_board = [row for row in self.board if any(cell == -1 for cell in row)]
        lines = ROWS - len(new_board)
        for _ in range(lines):
            new_board.insert(0, [-1 for _ in range(COLS)])
        self.board = new_board
        if lines > 0:
            self.lines_cleared += lines
            self.score += SCORES.get(lines, 0) * self.level
            new_level = (self.lines_cleared // 10) + 1
            if new_level != self.level:
                self.level = new_level
                self.drop_interval = self.compute_drop_interval()

    def spawn_piece(self):
        self.current = self.next
        self.next = Piece()
        self.current.x = (COLS // 2) - (len(self.current.shape[0]) // 2)
        self.current.y = 0
        if not self.can_place(self.current.shape, self.current.x, self.current.y):
            self.game_over = True

    def hard_drop(self):
        while self.can_place(self.current.shape, self.current.x, self.current.y + 1):
            self.current.y += 1
        self.place_piece()
        self.remove_full_lines()
        self.spawn_piece()

    def update(self, dt, commands):
        if self.game_over or self.paused:
            return

        moved = False
        rotated = False


        if commands.get('left'):
            if self.can_place(self.current.shape, self.current.x - 1, self.current.y):
                self.current.x -= 1
                moved = True
        if commands.get('right'):
            if self.can_place(self.current.shape, self.current.x + 1, self.current.y):
                self.current.x += 1
                moved = True

        if commands.get('rotate'):
            old_shape = [row[:] for row in self.current.shape]
            old_x = self.current.x
            self.current.rotate()
            kicked = False
            for dx in (0, -1, 1, -2, 2):
                if self.can_place(self.current.shape, self.current.x + dx, self.current.y):
                    self.current.x += dx
                    kicked = True
                    break
            if not kicked:
                self.current.shape = old_shape
                self.current.x = old_x
            else:
                rotated = True

        speed_multiplier = 10 if commands.get('soft_drop') else 1
        can_fall = self.can_place(self.current.shape, self.current.x, self.current.y + 1)

        if can_fall:
            self.drop_timer += dt * speed_multiplier
            if self.drop_timer >= self.drop_interval:
                self.drop_timer = 0.0
                self.current.y += 1
                self.landed = False
                self.lock_timer = 0.0
        else:
            if not self.landed:
                self.landed = True
                self.lock_timer = 0.0
            else:
                self.lock_timer += dt
                if self.lock_timer >= self.lock_delay:
                    self.place_piece()
                    self.remove_full_lines()
                    self.spawn_piece()
                    self.landed = False
                    self.lock_timer = 0.0
                    self.drop_timer = 0.0

        if self.landed and (moved or rotated):
            self.lock_timer = 0.0

# ---- DRAWING ----
def draw_board(screen, game, offset_x=0, offset_y=0):
    surf = pygame.Surface((CELL_SIZE*COLS, CELL_SIZE*ROWS))
    surf.fill(BLACK)
    for r in range(ROWS):
        for c in range(COLS):
            cell = game.board[r][c]
            rect = pygame.Rect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surf, GRAY, rect, 1)
            if cell != -1:
                color = COLORS[cell]
                inner = rect.inflate(-2, -2)
                pygame.draw.rect(surf, color, inner)
    p = game.current
    for r, row in enumerate(p.shape):
        for c, cell in enumerate(row):
            if cell:
                bx = p.x + c
                by = p.y + r
                if by >= 0:
                    rect = pygame.Rect(bx*CELL_SIZE, by*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    inner = rect.inflate(-2, -2)
                    pygame.draw.rect(surf, p.color, inner)
                    pygame.draw.rect(surf, WHITE, rect, 1)
    screen.blit(surf, (offset_x, offset_y))

HUD_TEXT = (255, 255, 0)        # yellow for text

# Define soft retro colors
SIDEBAR_BG = (40, 40, 80)  # full right side background
TEXT_COLOR = (255, 255, 0) # for labels
SUBTEXT_COLOR = (200, 200, 200) # for highscores

PANEL_WIDTH = 300  # width of right sidebar

def draw_next(screen, next_piece, x, y):

    font = RETRO_FONT_TITLE   # use retro font
    # font = pygame.font.Font(pygame.font.match_font('freesansbold'), 20)
    pygame.draw.rect(screen, SIDEBAR_BG, (x-10, y-10, 120, 150), border_radius=5)  # changed color
    label = font.render("NEXT", True, HUD_TEXT)
    screen.blit(label, (x, y))
    px = x
    py = y + 30
    for r, row in enumerate(next_piece.shape):
        for c, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(px + c*CELL_SIZE, py + r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                inner = rect.inflate(-2, -2)
                pygame.draw.rect(screen, next_piece.color, inner)
                pygame.draw.rect(screen, WHITE, rect, 1)

def draw_hud(screen, game, x, y):

    font = RETRO_FONT_TITLE  # use retro font
    # font = pygame.font.Font(pygame.font.match_font('freesansbold'), 20)
    pygame.draw.rect(screen, SIDEBAR_BG, (x-10, y-10, 180, 120), border_radius=5)  # changed color
    lines = [
        f"Score: {game.score}",
        f"Level: {game.level}",
        f"Lines: {game.lines_cleared}"
    ]
    for i, l in enumerate(lines):
        txt = font.render(l, True, HUD_TEXT)
        screen.blit(txt, (x, y + i*30))

def draw_highscores(screen, x, y):

    font_label = RETRO_FONT_TITLE
    font_scores = RETRO_FONT_SMALL
    # font = pygame.font.Font(pygame.font.match_font('freesansbold'), 18)
    scores = load_highscores()
    pygame.draw.rect(screen, SIDEBAR_BG, (x-10, y-10, 180, 200), border_radius=5)  # changed color
    label = font_label.render("TOP 5", True, HUD_TEXT)
    screen.blit(label, (x, y))
    for i, s in enumerate(scores):
        txt = font_scores.render(f"{i+1}. {s['name']} - {s['score']}", True, (200,200,200))
        screen.blit(txt, (x, y + 28*(i+1)))



def draw_sidebar_background(screen):
    WIDTH, HEIGHT = screen.get_size()
    panel_rect = pygame.Rect(WIDTH - PANEL_WIDTH, 0, PANEL_WIDTH, HEIGHT)
    pygame.draw.rect(screen, SIDEBAR_BG, panel_rect)



# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Učitavanje tvog sklearn klasifikatora
classifier = joblib.load("../evaluation_results_multi_cv_all/MLP/mlp_model.pkl")  # promeni putanju ako treba
scaler = joblib.load("../evaluation_results_multi_cv_all/feature_scaler.pkl")
label_encoder = joblib.load("../evaluation_results_multi_cv_all/label_encoder.pkl")


y_history = deque(maxlen=10)
time_history = deque(maxlen=10)
angle_history = deque(maxlen=10)
points = deque(maxlen=10)  # za crtanje trajektorije

COOLDOWNS = {
    'left': 0.3,       # seconds
    'right': 0.3,
    'rotate': 0.5,
    'soft_drop': 0.1,
    'hard_drop': 0.5
}

last_command_times = {
    'left': 0,
    'right': 0,
    'rotate': 0,
    'soft_drop': 0,
    'hard_drop': 0
}

# SWIPE_THRESHOLD = 0.20
SPEED_THRESHOLD = 0.3
last_hand_label = None

# globals
rotation_active = False
neutral_reached = True


# Folder for screenshots
SAVE_DIR = "gesture_screenshots"
os.makedirs(SAVE_DIR, exist_ok=True)


def capture_screenshots(gesture_name, num=10, delay=0.1):
    """
    Capture `num` screenshots when a gesture is recognized.
    delay = time between screenshots (in seconds)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    gesture_dir = os.path.join(SAVE_DIR, f"{gesture_name}_{timestamp}")
    os.makedirs(gesture_dir, exist_ok=True)

    print(f"📸 Capturing {num} screenshots for '{gesture_name}' gesture...")

    for i in range(num):
        filename = os.path.join(gesture_dir, f"{gesture_name}_{i + 1:03d}.png")
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        time.sleep(delay)

    print(f"✅ Saved {num} screenshots in {gesture_dir}")

gesture_text = "Neutral"


def get_hand_commands(result, frame, update):
    current_key_pressed = set()
    global last_hand_label, rotation_active, neutral_reached, gesture_text

    h, w, _ = frame.shape

    commands = {'left': False, 'right': False, 'pause':False, 'rotate': False, 'soft_drop': False, 'hard_drop': False}

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # --- CRTAJ LANDMARKS ZA DEBUG ---
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- EXTRACT FEATURES ZA CLASSIFIER ---
        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x)
            features.append(lm.y)
        features = np.array(features).reshape(1, -1)

        # --- PREDIKCIJA GESTA ---
        pred_scaled = scaler.transform(features)
        probs = classifier.predict_proba(pred_scaled)[0]
        confidence = np.max(probs)
        pred_num = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_num])[0]

        gesture_text = "Neutral"

        last_command = None

        if pred_label =='open_hand' and confidence > 0.99:
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

            y_history.append(wrist.y)
            time_history.append(time.time())
            points.append((wrist_x, wrist_y))

            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)
                cv2.circle(frame, points[i], 4, (0, 255, 0), -1)


            # Use pinky-tip to measure overall hand twist
            wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
            pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
            index_base = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]

            # Vector across hand base (wrist → index_base) or (wrist → pinky_tip)
            dx = pinky_tip.x - index_base.x
            dy = pinky_tip.y - index_base.y

            angle = math.degrees(math.atan2(dy, dx))
            angle_history.append(angle)


            if len(angle_history) > 5:
                angle_history.popleft()

            if len(angle_history) >= 5:
                avg_start = np.mean(list(angle_history)[:2])
                avg_end = np.mean(list(angle_history)[-2:])
                angle_diff = (avg_end - avg_start + 180) % 360 - 180

                rotation_threshold = 20
                neutral_threshold = 10
                angle_abs = abs(angle_diff)

                if angle_diff > rotation_threshold and not rotation_active and neutral_reached:
                    if update and last_hand_label == 'open_hand':
                        commands['rotate'] = True
                        rotation_active = True
                        neutral_reached = False
                        gesture_text = "Rotate"


                if angle_abs < neutral_threshold:
                    rotation_active = False
                    neutral_reached = True

            if len(y_history) == y_history.maxlen:
                diff = y_history[0] - y_history[-1]
                dt = time_history[-1] - time_history[0]

                if dt > 0:
                    speed = diff / dt  # brzina promene y-ose
                else:
                    speed = 0


                if speed < -SPEED_THRESHOLD :
                    if update and last_command != 'hard_drop': # and now - last_command_times['hard_drop'] > COOLDOWNS['hard_drop']:
                        commands['hard_drop'] = True
                        gesture_text = "Hard Drop"

                        # reset motion to avoid double trigger
                        y_history.clear()
                        angle_history.clear()

            gesture_text = "Open Hand"
        else:
            y_history.clear()
            angle_history.clear()

        if pred_label == 'thumb_left' and confidence > 0.99:
            if update:
                commands['left'] = True
                gesture_text = "Left"
        elif pred_label == 'thumb_right' and confidence > 0.99:
            if update:
                commands['right'] = True
                gesture_text = "Right"

        elif pred_label == 'fist' and confidence > 0.99:
            gesture_text = "Fist"

        elif pred_label == 'thumb_down' and confidence > 0.99:
            if update:
                commands['soft_drop'] = True
                gesture_text = "Thumb Down"



        last_hand_label = pred_label



    return commands

rotation_active = False
neutral_reached = True

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow('Hand', cv2.WINDOW_NORMAL)
scale_factor = 2.0

ret, frame = cap.read()
# ---- MAIN ----
def main():


    global RETRO_FONT_SMALL, RETRO_FONT_TITLE
    pygame.init()
    try:
        RETRO_FONT_TITLE = pygame.font.Font("PressStart2P.ttf", 24)
        RETRO_FONT_SMALL = pygame.font.Font("PressStart2P.ttf", 18)
    except:
        # fallback if TTF not found
        RETRO_FONT_TITLE = pygame.font.SysFont("Courier New", 24, bold=True)
        RETRO_FONT_SMALL = pygame.font.SysFont("Courier New", 18, bold=True)

    screen = pygame.display.set_mode((WIDTH + PANEL_WIDTH, HEIGHT))
    pygame.display.set_caption("Tetris - Python (pygame)")
    clock = pygame.time.Clock()
    game = Tetris()

    rotate_cooldown = 0.15
    hard_drop_cooldown = 0.2
    hard_drop_timer = 0.0
    rotate_timer = 0.0
    move_cooldown = 0.08
    move_timer = 0.0

    frame_counter = 0

    running = True
    # Load retro font at the start of the game

    while running:
        dt = clock.tick(FPS) / 1000.0
        rotate_timer += dt
        move_timer += dt
        hard_drop_timer += dt




        ret, frame = cap.read()
        frame_counter += 1

        # Convert frame to RGB and process only ovo
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

        commands = {'left': False, 'right': False,'pause': False, 'rotate': False, 'soft_drop': False, 'hard_drop': False}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()
                pygame.quit()
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_p:
                    game.paused = not game.paused
                if event.key == pygame.K_r:
                    game = Tetris()
        commands = get_hand_commands(result, frame, True)

        if commands['rotate']:
            if rotate_timer >= rotate_cooldown:
                rotate_timer = 0.0
            else:
                commands['rotate'] = False

        elif commands['left']:
            if move_timer >= move_cooldown:
                move_timer = 0.0
            else:
                commands['left'] = False
        elif commands['right']:
            if move_timer >= move_cooldown:
                move_timer = 0.0
            else:
                commands['right'] = False
        elif commands['hard_drop']:
            if hard_drop_timer >= hard_drop_cooldown:
                hard_drop_timer= 0.0
            else:
                commands['hard_drop'] = False

        if commands['hard_drop']:
            if not game.game_over and not game.paused:
                game.hard_drop()

        game.update(dt, commands)
        display_frame = cv2.flip(frame, 1)
        cv2.putText(display_frame, gesture_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand', display_frame)

        screen.fill((10,10,10))
        draw_board(screen, game, 0, 0)
        # draw the game board
        draw_sidebar_background(screen)
        draw_next(screen, game.next, WIDTH + 20, 20)
        draw_hud(screen, game, WIDTH + 20, 160)
        draw_highscores(screen, WIDTH + 20, 260)

        font = pygame.font.SysFont(None, 48)
        if game.game_over:
            over = RETRO_FONT_TITLE.render("GAME OVER", True, (200, 30, 30))
            screen.blit(over, (WIDTH//2 - over.get_width()//2, HEIGHT//2 - 40))
            sub = RETRO_FONT_SMALL.render("Press R to restart", True, WHITE)
            screen.blit(sub, (WIDTH//2 - sub.get_width()//2, HEIGHT//2 + 20))
            # update highscores once per game over
            if not hasattr(game, "highscores_done"):
                update_highscores(game.score, screen)
                game.highscores_done = True
        elif game.paused:
            paused = font.render("PAUSED", True, (200, 200, 30))
            screen.blit(paused, (WIDTH//2 - paused.get_width()//2, HEIGHT//2 - 20))

        small = pygame.font.SysFont(None, 20)
        hints = [
            "Controls:",
            "Left - Thumb Left",
            "Right - Thumb Right",
            "Up - Rotate Hand",
            "Down - Thumb Down",
            "Space - Hand Up->Down",
            "P - pause",
            "R - restart",
            "Esc - quit"
        ]
        for i, h in enumerate(hints):
            txt = RETRO_FONT_SMALL.render(h, True, WHITE)
            screen.blit(txt, (WIDTH + 20, HEIGHT - 20*(len(hints)-i)))

        pygame.display.flip()

    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()