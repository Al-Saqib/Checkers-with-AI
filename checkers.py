# Import Statements 
import random
import tkinter as tk
from tkinter import messagebox, Toplevel, Label

# Class to encacpsulate the game
class Checkers_Game:
    
    grey = 1  # Represents the grey player
    red = 0  # Represents the red player
    grey_soldier = 1  # Grey soldier piece
    grey_king = 3  # Grey king piece
    red_soldier = 2  # Red soldier piece
    red_king = 4  # Red king piece
    pos_mov_x = [1, 1, -1, -1]  # x-axis movement possibilities
    pos_mov_y = [1, -1, 1, -1]  # y-axis movement possibilities
    
    def __init__(self, size=8):
        self.size = size
        if self.size % 2 != 0 or self.size < 8:
            raise ValueError("Invalid board size")
        self.board = self.initialize_board()
        
    # Method to initialize the board
    def initialize_board(self):
        """
        Initializes the game board with the starting positions of the pieces.
        """
        board = []
        check_pie = self.grey_soldier
        mid = self.size // 2
        # Iterate over each row of the board to set the initial piece positions
        for i in range(self.size):
            row = []
            if i < mid - 1:
                check_pie = self.grey_soldier
            elif i == mid - 1 or i == mid:
                check_pie = 0
            else:
                check_pie = self.red_soldier
            # Iterate over each cell in the row to set the piece values
            for j in range(self.size):
                if (i + j) % 2 == 1:
                    row.append(check_pie)
                else:
                    row.append(0)
            board.append(row)
        return board
    
    
    
    # Method to copy the board
    def copy_board(self):
        return [row[:] for row in self.board] 
    
    # Method to validate a move
    def val_move(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size # Check if the move is within the board boundaries
    
    
    def successor_pos(self, x, y):
        # Check if the current position is empty and return empty lists if true
        if self.board[x][y] == 0:
            return [], []

        # Determine player and piece type
        player = self.board[x][y] % 2
        is_soldier = self.board[x][y] in [self.grey_soldier, self.red_soldier]
        direction = 1 if player == self.grey else -1
        max_distance = 2 if is_soldier else 4

        # Prepare lists for normal and capturing moves
        norm_movs, cap_movs = [], []

        # Iterate over possible moves within the allowed distance
        for i in range(max_distance):
            fol_x, fol_y = x + direction * self.pos_mov_x[i], y + direction * self.pos_mov_y[i]
            if self.val_move(fol_x, fol_y):
                if self.board[fol_x][fol_y] == 0:
                    norm_movs.append((fol_x, fol_y))
                elif self.board[fol_x][fol_y] % 2 != player:
                    # Check for a valid capture move
                    capture_x, capture_y = fol_x + direction * self.pos_mov_x[i], fol_y + direction * self.pos_mov_y[i]
                    if self.val_move(capture_x, capture_y) and self.board[capture_x][capture_y] == 0:
                        cap_movs.append((capture_x, capture_y))

        return norm_movs, cap_movs

    
    # Method to find moves for a specific piece
    def find_moves_for_piece(self, x, y, player):
        """ Helper method to find moves for a specific piece. """
        if self.board[x][y] != 0 and self.board[x][y] % 2 == player:
            norm, cap = self.successor_pos(x, y)
            return norm, cap
        return None, None

    def successor_mov(self, player):
        cap_movs = []
        norm_movs = []

        # Scan the board for pieces of the current player and collect possible moves
        for x in range(self.size):
            for y in range(self.size):
                norm, cap = self.find_moves_for_piece(x, y, player)
                if norm:
                    norm_movs.append(((x, y), norm))
                if cap:
                    cap_movs.append(((x, y), cap))

        # Prefer capture moves if available, otherwise return normal moves
        return cap_movs if cap_movs else norm_movs

    
    # Method to execute a move
    def execute_mov(self, x, y, fol_x, fol_y):
        self.board[fol_x][fol_y] = self.board[x][y]
        self.board[x][y] = 0
        removed = 0 # Track the piece removed during a capture
        promote = False # Track if the move results in a promotion
        continued_capture = False # Track if the move allows for further captures
        captured_king = False  # Track if a king was captured

        # Calculate if the move is a capture
        if abs(fol_x - x) == 2:
            dx = (fol_x - x) // 2
            dy = (fol_y - y) // 2
            mid_x, mid_y = x + dx, y + dy
            removed = self.board[mid_x][mid_y]
            self.board[mid_x][mid_y] = 0
            continued_capture = any(self.successor_pos(fol_x, fol_y)[1])  # Check for further captures

            # Check if a king was captured and the capturing piece is not a king
            if removed in [self.grey_king, self.red_king] and self.board[fol_x][fol_y] in [self.grey_soldier, self.red_soldier]:
                captured_king = True

        # Handle promotions
        # Promote grey soldier to grey king if it reaches the opposite end of the board
        if self.board[fol_x][fol_y] == self.grey_soldier and fol_x == self.size - 1:
            self.board[fol_x][fol_y] = self.grey_king
            promote = True
        # Promote red soldier to red king if it reaches the opposite end of the board
        elif self.board[fol_x][fol_y] == self.red_soldier and fol_x == 0:
            self.board[fol_x][fol_y] = self.red_king
            promote = True
        elif captured_king:  # Promote if a king was captured
            if self.board[fol_x][fol_y] == self.grey_soldier:
                self.board[fol_x][fol_y] = self.grey_king
            elif self.board[fol_x][fol_y] == self.red_soldier:
                self.board[fol_x][fol_y] = self.red_king
            promote = True
            continued_capture = False  # End the turn after regicide

        # Return status of the move: continue capture, piece removed, was promotion
        return continued_capture, removed, promote


    
    # Method to reverse a move
    def reverse_mov(self, x, y, fol_x, fol_y, removed=0, promoted=False):
        # Reverse the move by moving the piece back to its original position
        if promoted:
            if self.board[fol_x][fol_y] == self.grey_king:
                self.board[fol_x][fol_y] = self.grey_soldier
            elif self.board[fol_x][fol_y] == self.red_king:
                self.board[fol_x][fol_y] = self.red_soldier
        # Move the piece back to its original position
        self.board[x][y] = self.board[fol_x][fol_y]
        self.board[fol_x][fol_y] = 0
        # Reverse the capture if applicable
        if abs(fol_x - x) == 2:
            dx = (fol_x - x) // 2
            dy = (fol_y - y) // 2
            mid_x, mid_y = x + dx, y + dy
            self.board[mid_x][mid_y] = removed

    
# Class to implement the AI    
class Checkers_AI:
    
    
    infinity = float('inf')  # Represents positive infinity
    
    # Constructor to initialize the AI with the game
    def __init__(self, game):
        self.game = game
        self.stateCounter = {}
    
    
    def evaluate(self, maximizer):
        """
        Evaluates the current board state from the perspective of the maximizer player.
        A positive value favors the maximizer, while a negative value favors the minimizer.
        """
        value = 0
        board = self.game.copy_board()
        size = self.game.size
        for i in range(size):
            for j in range(size):
                piece = board[i][j]
                if piece != 0:
                    # Calculate the base score for men and kings
                    piece_value = (piece + 1) // 2
                    # Determine if current piece is maximizer's or minimizer's
                    if piece % 2 == maximizer:
                        value += piece_value
                    else:
                        value -= piece_value
        return value * 120  # Scale the score to make it more significant
    
    
    def evaluate_adv(self):
        """
        Calculate a unique encoding of the board state that combines the position
        and type of each piece, giving a more positional-value-based score.

        """
        value = 0
        board = self.game.copy_board()  # Retrieve the current board state from Checkers_Game
        size = self.game.size  # The size of the board

        # Iterate over each cell in the board to calculate its contribution to the board encoding
        for i in range(size):
            for j in range(size):
                piece = board[i][j]
                if piece != 0:
                    num = i * size + j + 5  # Compute a unique position number for each cell
                    value += num * piece  # Multiply the piece value by its unique position number
                    
        print("Encoding the value: ", self.game.copy_board(), self.game.size, value)

        return value
    
    # Method to calculate the state value
    def state_val(self, maximizer):
        max_pie = 0
        min_pie = 0
        board = self.game.copy_board()
        size = self.game.size
        # Iterate over each cell in the board to count the number of pieces for each player
        for i in range(size):
            for j in range(size):
                piece = board[i][j]
                if piece != 0:
                    if piece % 2 == maximizer:
                        max_pie += 1
                    else:
                        min_pie -= 1

        eval_key = self.evaluate_adv()
        # Only access the current count, do not increment here
        state_count = self.stateCounter.get(eval_key, 0)

        # Return negative state count if maximizer has more pieces, otherwise return 0
        return -state_count if max_pie > min_pie else 0
    
    # Method to implement the Minimax algorithm
    def min_max_algo(self, player, maximizer, depth=0, alpha=-float('inf'), beta=float('inf'), maxDepth=4, evaluate=None, movs=None):
        # Check if the game has ended or the maximum depth has been reached
        if movs is None:
            movs = self.game.successor_mov(player)
        if not movs or depth == maxDepth:
            val = self.evaluate(maximizer)  # Assuming evaluate is a method of Checkers_AI
            if val < 0:
                val += depth
            return val
        
        bVal = -self.infinity if player == maximizer else self.infinity # Initialize the best value based on the player
        
        movs.sort(key=lambda mov: len(mov[1]))  # Sorting moves based on the length, if applicable
        # Iterate over each possible move to determine the best move
        for mov in movs:
            x, y = mov[0]
            for fol_x, fol_y in mov[1]: # Iterate over each possible next position
                canCaptureAgain, removed, promoted = self.game.execute_mov(x, y, fol_x, fol_y) # Execute the move
                played = False # Track if a move was played
                if canCaptureAgain:
                    _, folCap = self.game.successor_pos(fol_x, fol_y)
                    if folCap:
                        played = True # A move was played
                        nmovs = [((fol_x, fol_y), folCap)] # Prepare the next moves
                        # Recursively call the min_max_algo method to determine the best move
                        if player == maximizer:
                            eval = self.min_max_algo(player, maximizer, depth + 1, alpha, beta, maxDepth, evaluate, nmovs)
                            bVal = max(bVal, eval)
                            alpha = max(alpha, bVal)
                        else:
                            eval = self.min_max_algo(player, maximizer, depth + 1, alpha, beta, maxDepth, evaluate, nmovs)
                            bVal = min(bVal, eval)
                            beta = min(beta, bVal)
                if not played: # If no move was played, calculate the best move
                    if player == maximizer:
                        eval = self.min_max_algo(1 - player, maximizer, depth + 1, alpha, beta, maxDepth, evaluate)
                        bVal = max(bVal, eval)
                        alpha = max(alpha, bVal)
                    else:
                        eval = self.min_max_algo(1 - player, maximizer, depth + 1, alpha, beta, maxDepth, evaluate)
                        bVal = min(bVal, eval)
                        beta = min(beta, bVal)
                self.game.reverse_mov(x, y, fol_x, fol_y, removed, promoted)
                if beta <= alpha:
                    break
            if beta <= alpha:
                break
        return bVal
        
    # Method to implement the Alpha-Beta Pruning algorithm
    def a_b_pruning(self, player, movs=None, maxDepth=4, evaluate=None, printable=True):
        
        print(f"Starting alpha-beta pruning with maxDepth: {maxDepth}")
        # Check if the game has ended or the maximum depth has been reached
        if movs is None:
            movs = self.game.successor_mov(player)
        if not movs:
            if printable:
                print(("Grey" if player == self.game.red else "Red") + " Player has won the game!")
            return False, False
        
        self.stateCounter[self.evaluate_adv()] = self.stateCounter.get(self.evaluate_adv(), 0) + 1 # Increment the state counter
        random.shuffle(movs) # Shuffle the moves to randomize the order
        bVal = -float('inf') # Initialize the best value
        bMov = None # Initialize the best move
        # Iterate over each possible move to determine the best move
        for move in movs:
            print("Moves structure:", movs)
            x, y = move[0]
            for fol_x, fol_y in move[1]: # Iterate over each possible next position
                _, removed, promoted = self.game.execute_mov(x, y, fol_x, fol_y)
                val = self.min_max_algo(1 - player, player, maxDepth=maxDepth,evaluate=self.evaluate)
                val += 2 * self.state_val(player)
                self.game.reverse_mov(x, y, fol_x, fol_y, removed, promoted)
                if val > bVal: # Update the best value and move if a better move is found
                    bVal = val
                    bMov = (x, y, fol_x, fol_y)
        # Execute the best move            
        x, y, fol_x, fol_y = bMov
        print("Best Value: ", bVal)
        print("Best Move: ", (x, y), (fol_x, fol_y))
        CanCaptureAgain, removed, _ = self.game.execute_mov(x, y, fol_x, fol_y)
        if CanCaptureAgain: # Check if the move allows for further captures
            _, caps = self.game.successor_pos(fol_x, fol_y)
            if caps: # If further captures are possible, continue the capture
                self.a_b_pruning(player, [((fol_x, fol_y), caps)], maxDepth, evaluate, printable)
        self.stateCounter[self.evaluate_adv()] = self.stateCounter.get(self.evaluate_adv(), 0) + 1 # Increment the state counter
        reset = removed != 0 # Check if the move resulted in a capture
        return True, reset


         
# Global variables
# g_mode = solo_play
# max_dep = 4
size_checker = 8 # Size of the checkers board
player_first = Checkers_Game.red # Player to start the game
depth_ascend = False # Flag to indicate if the depth should increase
solo_play = 0 # Flag to indicate solo play
multi_play = 1 # Flag to indicate multiplayer
min_max = 0 # Flag to indicate Minimax algorithm
image_size = 80 # Size of the images on the board


# Class to implement the GUI
class GraphicalUserInterface:
    # Constructor to initialize the GUI
    def __init__(self, master):
        self.master = master
        self.master.title("Checkers Tavern")
        self.master.withdraw()
        self.initial_setup()
    # Method to set up the game    
    def initial_setup(self):
        self.setup_window = tk.Toplevel(self.master)
        self.setup_window.title("Game Setup")

        # Select Game Mode
        tk.Label(self.setup_window, text="Select Game Mode:").pack(pady=5) # Add a label to the setup window
        tk.Button(self.setup_window, text="Player vs AI", command=lambda: self.select_game_mode(solo_play)).pack(pady=5) # Add a button to select Player vs AI
        tk.Button(self.setup_window, text="Player vs Player", command=lambda: self.select_game_mode(multi_play)).pack(pady=5) # Add a button to select Player vs Player
        
        # button to display checker rules 
        tk.Button(self.setup_window, text="Show Rules", command=self.show_rules).pack(pady=10) # Add a button to show the rules of the game
        
    
    def select_game_mode(self, mode):
        global g_mode
        g_mode = mode
        if g_mode == solo_play:
            self.select_difficulty()  # Proceed to difficulty selection for AI games
        else:
            self.setup_window.destroy()
            self.master.deiconify()  # Directly show the main game window for player vs player
            self.start_game(None)  # Directly start the game without setting a difficulty
        

    def select_difficulty(self): # Method to select the AI difficulty
        self.difficulty_window = tk.Toplevel(self.master) 
        self.difficulty_window.title("Select Difficulty")

        tk.Label(self.difficulty_window, text="Select Difficulty:").pack(pady=5) # Add a label to the difficulty selection window
        tk.Button(self.difficulty_window, text="Easy", command=lambda: self.start_game(2)).pack(pady=5) 
        tk.Button(self.difficulty_window, text="Medium", command=lambda: self.start_game(4)).pack(pady=5)
        tk.Button(self.difficulty_window, text="Hard", command=lambda: self.start_game(6)).pack(pady=5)
        
        
    def show_rules(self):
        # This method creates a new window to display the rules.
        rules_window = Toplevel(self.master)
        rules_window.title("Rules of Checkers")
        rules_text = """Rules of Checkers:
        - Each player starts with 12 pieces placed on the dark squares of the board closest to them.
        - Pieces move diagonally and can only move forward until they become kings.
        - Pieces only move one square at a time to an adjacent unoccupied dark square.
        - To capture an opponent's piece, jump over it to an open square.
        - If a piece reaches the opposite side of the board, it is crowned and becomes a king.
        - If a piece captures a king, the piece becomes a king.
        - Kings can move and capture both forward and backward.
        - The player with no available moves loses the game."""
        Label(rules_window, text=rules_text, justify=tk.LEFT, padx=10, pady=10).pack()

        # a close button to the rules window
        tk.Button(rules_window, text="Close", command=rules_window.destroy).pack(pady=10)
        
        
    def show_help(self):
        """
        Displays a pop-up window with instructions on how to interact with the game UI.
        """
        help_window = Toplevel(self.master)
        help_window.title("How to Play")
        instructions = """How to Play:
        - Click on a piece to select it. Valid moves will be highlighted.
        - Click on a highlighted square to move the selected piece to that square.
        - Click on the piece again to deselect it. Pieces that can be moved will be highlighted.
        - If a capture is possible, you must make the capture. You will not be able to make other moves.
        - The game automatically alternates between players after a valid move.
        - If multiple captures are possible, the game will prompt you to continue capturing."""
        Label(help_window, text=instructions, justify=tk.LEFT, padx=10, pady=10).pack()

        # close button to the help window
        tk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)

    # Method to start the game
    def start_game(self, maxDepth):
        self.maxDepth = maxDepth  # This will be None for Player vs Player
        if self.maxDepth is not None:  # Check if it's an AI game and difficulty was set
            self.difficulty_window.destroy()
        self.setup_window.destroy()
        self.master.deiconify()
        self.initialize_game()

    def initialize_game(self):
        self.game = Checkers_Game(size_checker) # Initialize the game
        self.prev_board = self.game.copy_board() # Store the previous board state
        self.prev_pointer = 0 # Pointer to the previous board state
        
        self.prev_x = None # Initialize the previous x-coordinate
        self.prev_y = None # Initialize the previous y-coordinate
        self.to_cap = False # Initialize the capture flag
        self.count = 0 # Initialize the move count

        self.player = player_first
        if g_mode == solo_play:
            self.ai = Checkers_AI(self.game)  # Initialize AI only for solo play
            if self.player == Checkers_Game.grey and g_mode == solo_play:
                self.ai.a_b_pruning(1 - self.player, maxDepth=self.maxDepth, evaluate=self.ai.evaluate, printable=False) # change to max_depth if you want to use a fixed depth
                self.prev_board = self.game.copy_board()
        else:
            self.ai = None  # No AI for player vs player
        self.setup_game()
        

    def setup_game(self):
        # Setting up game UI elements (boards, pieces, etc.)
        self.game_board = [[None] * self.game.size for _ in range(self.game.size)]
        self.init_board_gui()
        
        # Add a Help button to the game UI
        tk.Button(self.master, text="Help", command=self.show_help).pack(pady=5)
        
    
    # Method to initialize the game board
    def init_board_gui(self):
        
        b_fram = tk.Frame(master=self.master) # Create a frame for the game board
        b_fram.pack(fill=tk.BOTH, expand=True)
        for i in range(self.game.size): # Iterate over each row of the board
            b_fram.columnconfigure(i, weight=1, minsize=image_size)
            b_fram.rowconfigure(i, weight=1, minsize=image_size)
            
            for j in range(self.game.size): # Iterate over each cell in the row
                fram = tk.Frame(master=b_fram)
                fram.grid(row=i, column=j, sticky="nsew")
                
                canvas = tk.Canvas(master=fram, width=image_size, height=image_size, bg="white") # Create a canvas for the cell
                canvas.pack(expand=True, fill=tk.BOTH)
                
                oval_id = canvas.create_oval(10, 10, image_size - 10, image_size - 10, fill='', outline='')
                
                self.game_board[i][j] = canvas 
                self.game_board[i][j].oval_id = oval_id
                
                canvas.bind("<Button-1>", self.update_click)
                
        fram_op = tk.Frame(master=self.master)
        fram_op.pack(expand=False)
        fram_count = tk.Frame(master=self.master)
        fram_count.pack(expand=False)
        
        self.update_board() # Update the game board
        
        fol_pos = [move[0] for move in self.game.successor_mov(self.player)] # Get the possible moves for the current player
        self.highlight_mov(fol_pos) # Highlight the possible moves for the current player
        
    # Method to update the game board    
    def update_board(self):
        for  i in range(self.game.size):
            f = i % 2 == 1
            for j in range(self.game.size):
                canvas = self.game_board[i][j]
                oval_id = canvas.oval_id
                
                if f:
                    canvas['bg'] = '#0000FF'  # Dark square
                else:
                    canvas['bg'] = '#F0F0F0'  # Light square
                
                piece = self.game.board[i][j] # Get the piece at the current position
                color = ''
                if piece == Checkers_Game.red_soldier: 
                    color = 'red' # Red soldier piece
                elif piece == Checkers_Game.red_king:
                    color = 'dark red' # Red king piece
                elif piece == Checkers_Game.grey_soldier:
                    color = 'gray' # Grey soldier piece
                elif piece == Checkers_Game.grey_king:
                    color = 'black'  # Grey king piece
                
                canvas.itemconfig(oval_id, fill=color)
                f = not f
                
            window.update()
    
    # Method to highlight the possible moves
    def highlight_mov(self, moves):
        for x in range(self.game.size): # Iterate over each row of the board
            for y in range(self.game.size):
                canvas = self.game_board[x][y]
                def_bg = '#0000FF' if (x + y) % 2 == 1 else '#F0F0F0'
                canvas.config(bg=def_bg)
            
        
        for postion in moves: # Iterate over each possible move
            x, y = postion
            canvas = self.game_board[x][y]
            canvas.config(bg='green') # Highlight the possible move
            
    # Method to update the game board after a move
    def update_click(self, event):
        info = event.widget.master.grid_info()
        x, y = info['row'], info['column']
        if self.prev_x is None and self.prev_y is None: # Check if a piece has been selected
            movs = self.game.successor_mov(self.player) # Get the possible moves for the current player
            print("Moves: ", movs) # Print the possible moves
            discovered = (x, y) in [move[0] for move in movs] # Check if the selected piece can move
            
            if discovered: # If the selected piece can move, highlight the possible moves
                self.prev_x = x
                self.prev_y = y
                normal, capture = self.game.successor_pos(x, y)
                pos = normal if not capture else capture
                self.highlight_mov(pos)
            else:
                messagebox.showinfo("Invalid Move", "No piece at selected square or piece cannot move.")
                print("Invalid Move")
                
            return
        
        norm_Pos, cap_Pos = self.game.successor_pos(self.prev_x, self.prev_y) 
        pos = norm_Pos if not cap_Pos else cap_Pos # Get the possible moves for the selected piece
        print("Possible Positions: ", pos) # Print the possible moves
        
        
        # Check if the selected square is a valid destination for the piece
        if (x, y) not in pos: 
            print("Invalid Move")
            if not self.to_cap:
                self.prev_x = None
                self.prev_y = None
                next_positions = [move[0] for move in self.game.successor_mov(self.player)]
                self.highlight_mov(next_positions)
            
            messagebox.showinfo("Invalid Move", "Selected square is not a valid destination for the piece.")
            print("Invalid Move: Selected square is not a valid destination for the piece.")
            
            return
                
                
         
        canCaptureAgain, removed, _ = self.game.execute_mov(self.prev_x, self.prev_y, x, y) # Execute the move
        self.highlight_mov([]) # Remove the highlighted moves
        self.update_board() # Update the game board
        self.count += 1 # Increment the move count
        self.prev_x = None # Reset the previous x-coordinate
        self.prev_y = None  # Reset the previous y-coordinate
        self.to_cap = False # Reset the capture flag
        
        
        if removed != 0: # Check if a piece was removed during the move
            self.count = 0
        if canCaptureAgain: # Check if the move allows for further captures
            _, nextCaptures = self.game.successor_pos(x, y)
            if nextCaptures: # If further captures are possible, highlight the possible moves
                self.to_cap = True
                self.prev_x = x
                self.prev_y = y
                self.highlight_mov(nextCaptures)
                
                return
            
        # Check if the game has ended    
        if g_mode == solo_play: # Check if it's a solo play game
            go_on, go_back = True, False
            if min_max == min_max: # Check if the Minimax algorithm is being used
                evaluate = Checkers_AI.evaluate
                if self.count > 20: # Check if the move count is greater than 20
                    evaluate = Checkers_AI.evaluate
                    if depth_ascend:
                        self.maxDepth = 7
                else:
                    evaluate = Checkers_AI.evaluate # Use the evaluate method for the AI
                    self.maxDepth = self.maxDepth
        
                go_on, go_back = self.ai.a_b_pruning(1 - self.player, maxDepth=self.maxDepth, evaluate=evaluate, printable=False) # Execute the AI move
            
            self.count += 1
            
            if not go_on: # Check if the game has ended
                messagebox.showinfo(message = "You have won the game!", title="Game Over")
                window.destroy()
                return
            self.update_board()
            if go_back:
                self.count = 0
        
        else: 
            self.player = 1 - self.player  # Switch players for player vs player games
            
        if self.count >= 100: # Check if the move count is greater than 100
            messagebox.showinfo(message="Draw at this point!", title="Game Over")
            window.destroy()
            return
        
        # Check if the current player has no possible moves
        next_positions = [move[0] for move in self.game.successor_mov(self.player)] # Get the possible moves for the current player
        if not next_positions: # Check if the current player has no possible moves
            if g_mode == solo_play: 
                messagebox.showinfo(message="You have lost the game!", title="Game Over")
            else:
                winner = "Red" if self.player == Checkers_Game.red else "Grey"
                messagebox.showinfo(message=f"You have won the game!", title="Game Over")
            window.destroy()
            
        
        self.prev_board = self.prev_board[:self.prev_pointer + 1] # Update the previous board state
        self.prev_board.append(self.game.copy_board()) # Append the current board state to the previous board
        self.prev_pointer += 1 # Increment the previous board pointer

        
# Main method to run the game       
if __name__ == "__main__":
    window = tk.Tk()
    app = GraphicalUserInterface(window)
    window.mainloop() 
