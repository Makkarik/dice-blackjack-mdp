import pygame
import sys

# Dice pip (dot) positions for values 1..6, relative to the dice square.
# Each position is defined as (offset_x, offset_y) in "fraction of dice side."
# For example, (0.5, 0.5) is the center of the die.
DICE_PIPS = {
    1: [(0.5, 0.5)],
    2: [(0.25, 0.25), (0.75, 0.75)],
    3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
    4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
    5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5),
        (0.25, 0.75), (0.75, 0.75)],
    6: [(0.25, 0.25), (0.75, 0.25),
        (0.25, 0.5), (0.75, 0.5),
        (0.25, 0.75), (0.75, 0.75)],
}

def draw_die(screen, center_x, center_y, value, size=100):
    """
    Draws a single die at the given center (center_x, center_y)
    with a specified side size, labeled with 'value' (1..6).
    """
    # Die background
    left = center_x - size//2
    top = center_y - size//2
    pygame.draw.rect(screen, (255, 255, 255), (left, top, size, size), border_radius=10)
    pygame.draw.rect(screen, (0, 0, 0), (left, top, size, size), 2, border_radius=10)

    # Draw pips
    if value is None:
        # Draw a question mark in the center
        font = pygame.font.SysFont(None, size)
        text_surface = font.render("?", True, (0, 0, 0))
        tx = center_x - text_surface.get_width() // 2
        ty = center_y - text_surface.get_height() // 2
        screen.blit(text_surface, (tx, ty))
    else:
        # Each pip is a small circle. We'll compute its position relative to the die rect.
        pip_radius = size // 10
        if value not in DICE_PIPS:
            return  # Guard against invalid dice value
        for (fx, fy) in DICE_PIPS[value]:
            px = left + int(fx * size)
            py = top + int(fy * size)
            pygame.draw.circle(screen, (0, 0, 0), (px, py), pip_radius)

def val_or_zero(x):
        return x if x is not None else 0

def show_dice(dice_values):
    """
    Launches a Pygame window and shows the dice:
    dice_values = [player_dice1, player_dice2, dealer_dice1, dealer_dice2].
    """
    pygame.init()

    # Window size
    WIDTH, HEIGHT = 600, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dice Example")

    # We assume dice_values is exactly four integers in 1..6
    if len(dice_values) != 4:
        print("Please provide exactly 4 dice values.")
        pygame.quit()
        sys.exit()

    # Basic layout: 2 dice for the dealer on top row, 2 dice for the player on bottom row.
    # We'll place them in the center horizontally, spaced horizontally as well.
    # For example, top row y=100, bottom row y=300, and x positions around 200 & 400.
    # Adjust as needed.

    # Extract the dice
    p1, p2, d1, d2 = dice_values

    dealer_total = val_or_zero(d1) + val_or_zero(d2)
    player_total = val_or_zero(p1) + val_or_zero(p2)

    running = True
    while running:
        screen.fill((50, 150, 50))  # Greenish background

        # Define font
        font = pygame.font.SysFont(None, 48)

        # Title text (optional)
        dealer_text = font.render("Dealer", True, (255, 255, 255))
        player_text = font.render("Player", True, (255, 255, 255))
        screen.blit(dealer_text, (WIDTH//2 - dealer_text.get_width()//2, 20))
        screen.blit(player_text, (WIDTH//2 - player_text.get_width()//2, HEIGHT - 70))

        # Write total sum
        dealer_sum = font.render(f"{dealer_total}", True, (255, 255, 255))
        player_sum = font.render(f"{player_total}", True, (255, 255, 255))
        screen.blit(dealer_sum, (WIDTH//2 - 5, 110))
        screen.blit(player_sum, (WIDTH//2 - 5, HEIGHT - 140))

        # Draw the four dice
        # Dealer dice on top
        draw_die(screen, WIDTH//2 - 100, HEIGHT//2 - 75, d1, size=100)
        draw_die(screen, WIDTH//2 + 100, HEIGHT//2 - 75, d2, size=100)
        # Player dice on bottom
        draw_die(screen, WIDTH//2 - 100, HEIGHT//2 + 75, p1, size=100)
        draw_die(screen, WIDTH//2 + 100, HEIGHT//2 + 75, p2, size=100)

        pygame.display.flip()

        # Wait for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
