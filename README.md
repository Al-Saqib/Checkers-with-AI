# Checkers Game with Minimax Algorithm and Alpha-Beta Pruning

## Description

This project is an implementation of the classic Checkers game, enhanced with AI capabilities using the Minimax algorithm and Alpha-Beta Pruning. The game features a graphical user interface (GUI) created with Tkinter, allowing for interactive gameplay with adjustable AI difficulty levels.

## Installation

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- Tkinter (usually included with Python, but can be installed separately if needed)

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/your-repo-name.git
```

2. Navigate to the project directory:

```bash
cd your-repo-name
```

3. Run the Checkers game:

```bash
python checkers.py
```

## Project Structure
checkers.py: The main Python file containing the implementation of the Checkers game with the Minimax algorithm and Alpha-Beta Pruning.
report.pdf: The project report detailing the specifications, functionality, and challenges encountered during the development of the game.

## Features
### Gameplay
- Interactive Checkers Gameplay: The game features a user-friendly GUI for interactive gameplay.
- Adjustable AI Cleverness Levels: Players can choose from different AI difficulty levels (Easy, Medium, Hard).
  
Search Algorithm
State Representation: The game board is represented as a two-dimensional list.
AI Move Generation: The AI generates possible moves using a successor function.
Minimax Evaluation and Alpha-Beta Pruning: The AI uses these algorithms to evaluate potential moves efficiently.
Use of Heuristics: The AI employs heuristic evaluation to make strategic decisions quickly.


Validation of Moves
Invalid Moves by AI: The AI is prevented from making invalid moves.
Checking User Moves Validity: User moves are validated for compliance with Checkers rules.
Rejection of Invalid User Moves with Explanation: Invalid user moves are rejected with an explanation.
Forced Capture: The game enforces the rule of forced capture.


Other Features
Multi-Step Capture Moves: Both user and AI can execute multi-step capture moves.
King Conversion at Baseline: Pieces are converted to kings upon reaching the opposite end of the board.
Regicide: Capturing a king results in immediate piece upgrade.
Hint for Next Possible Moves: The game provides hints for the next possible moves.


Display-Specific Features
Board Representation on Screen: The game board is visually represented with alternating dark and light squares.
Updating Display After Moves: The display updates after each move, showing the new board state.
Helpful Instructions: The game includes a help feature with instructions on gameplay and rules.
Game Interaction: Players can interact with the game through a point-and-click interface.
Display of Rules: The game rules are displayed in a dedicated section accessible from the GUI.


## Contributing
If you would like to contribute to this project, please follow these guidelines:
- Fork the repository.
- Create a new branch with a descriptive name.
Make your changes and commit them with clear and concise messages.
Push your changes to your forked repository.
Create a pull request detailing the changes you have made.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For questions or inquiries, please contact Abdullah Al Saqib Majumder at [your-email@example.com].
