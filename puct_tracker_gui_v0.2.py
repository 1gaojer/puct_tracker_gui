import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import shutil

# Ensure matplotlib uses the TkAgg backend
import matplotlib
matplotlib.use("TkAgg")

# =====================================================================
# Backend Logic: TreeSearchManager Class
# Implements Algorithm 1 from the provided paper.
# =====================================================================

class TreeSearchManager:
    def __init__(self, c_puct=1.414):
        self.c_puct = c_puct
        # Use Pandas DataFrame for storing metrics and easy manipulation
        self.df = pd.DataFrame(columns=['ID', 'ParentID', 'ARI', 'V_u', 'RankScore', 'PUCT', 'Notes', 'Code'])
        self.next_id = 1
        self.N_total = 0

    def set_c_puct(self, c_puct):
        self.c_puct = float(c_puct)
        self._calculate_metrics()

    def add_node(self, parent_id, ari_score, notes="", code=""):
        node_id = self.next_id
        
        # Validation
        if parent_id != 0 and self.df[self.df['ID'] == parent_id].empty:
             # Ensure parent exists if it's not the root (0)
             raise ValueError(f"Parent ID {parent_id} does not exist.")
        if parent_id == 0 and not self.df.empty:
            # Ensure only one root node exists
            raise ValueError("Root node already exists.")

        self.next_id += 1

        # Add to DataFrame (Initialize V_u=1, Algorithm 1 Line 7)
        new_row = pd.DataFrame([{
            'ID': node_id,
            'ParentID': parent_id,
            'ARI': ari_score,
            'V_u': 1,
            'RankScore': np.nan,
            'PUCT': np.nan,
            'Notes': notes,
            'Code': code
        }])
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        # Perform backpropagation (Algorithm 1 Lines 9-11)
        if parent_id != 0:
            self._backpropagate(parent_id)
            
        # Recalculate metrics
        self._calculate_metrics()
        return node_id

    def _backpropagate(self, parent_id):
        """Updates V(u) for all ancestors starting from the parent."""
        current_id = parent_id
        while current_id != 0:
            # Increment V_u of the ancestor
            self.df.loc[self.df['ID'] == current_id, 'V_u'] += 1
            
            # Move up to the next ancestor
            parent_row = self.df[self.df['ID'] == current_id]
            if parent_row.empty: break
            current_id = parent_row['ParentID'].iloc[0]

    def _calculate_metrics(self):
        T = len(self.df) # Tree size |T|
        if T == 0:
            self.N_total = 0
            return

        # 1. Calculate N_total (Algorithm 1 Line 3: Sum of all V(u))
        self.N_total = self.df['V_u'].sum()
        
        # 2. Calculate RankScore
        if T == 1:
            self.df.loc[:, 'RankScore'] = 1.0
        else:
            # Rank from 1 (worst) to T (best)
            ranks = self.df['ARI'].rank(method='min')
            # Normalize: (Rank(u) - 1) / (T - 1)
            self.df.loc[:, 'RankScore'] = (ranks - 1) / (T - 1)

        # 3. Calculate PUCT (Algorithm 1 Line 4)
        P_T_u = 1 / T # Flat prior P_T(u)
        sqrt_N_total = np.sqrt(self.N_total)
        
        # PUCT(u) = RankScore(u) + C_puct * P_T(u) * (sqrt(N_total) / (1 + V(u)))
        exploration_term = self.c_puct * P_T_u * (sqrt_N_total / (1 + self.df['V_u']))
        self.df.loc[:, 'PUCT'] = self.df['RankScore'] + exploration_term

    def get_next_node_to_explore(self):
        # Select u = argmax(PUCT(u))
        if self.df.empty: return None
        return self.df.loc[self.df['PUCT'].idxmax()]

    def get_best_node(self):
        # Select u = argmax(TaskScore(u))
        if self.df.empty: return None
        return self.df.loc[self.df['ARI'].idxmax()]

    def build_graph(self):
        """Builds a NetworkX graph from the DataFrame for visualization."""
        G = nx.DiGraph()
        for index, row in self.df.iterrows():
            G.add_node(row['ID'], ARI=row['ARI'])
            if row['ParentID'] != 0:
                G.add_edge(row['ParentID'], row['ID'])
        return G

    def save_state(self, filepath='puct_state.csv'):
        # Ensure IDs are saved as integers
        df_save = self.df.copy()
        df_save['ID'] = df_save['ID'].astype(int)
        df_save['ParentID'] = df_save['ParentID'].astype(int)
        df_save['V_u'] = df_save['V_u'].astype(int)
        df_save.to_csv(filepath, index=False)

    def load_state(self, filepath='puct_state.csv'):
        if os.path.exists(filepath):
            self.df = pd.read_csv(filepath)
            # Ensure data types are correct and handle potential NaNs
            self.df['ID'] = self.df['ID'].astype(int)
            self.df['ParentID'] = self.df['ParentID'].astype(int)
            self.df['V_u'] = self.df['V_u'].astype(int)
            self.df['Code'] = self.df['Code'].fillna('')
            self.df['Notes'] = self.df['Notes'].fillna('')
            
            self.next_id = self.df['ID'].max() + 1 if not self.df.empty else 1
            self._calculate_metrics()
            return True
        return False

# =====================================================================
# GUI Application: PUCTTrackerApp Class
# =====================================================================

class PUCTTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Workflow PUCT Tracker")
        self.geometry("1300x850")
        self.style = ttk.Style()
        self.style.theme_use('clam') # Use a modern theme

        # Initialize the backend logic (C_puct=1.414 as requested)
        self.manager = TreeSearchManager(c_puct=1.414)
        self.state_file = 'puct_tracker_state.csv'

        # --- NEW: Undo/Redo Stacks ---
        self.undo_stack = []
        self.redo_stack = []

        self._create_widgets()
        self._load_initial_state()

        # --- NEW: Keyboard bindings for undo/redo ---
        self.bind_all("<Control-z>", self.undo_action)
        self.bind_all("<Control-y>", self.redo_action)

    def _create_widgets(self):
        # Use a PanedWindow for resizable panels
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Left Panel (Data Entry and Table)
        left_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(left_frame, weight=2)

        # Right Panel (Visualization and Details)
        right_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(right_frame, weight=1)

        # --- Widgets ---
        self._create_entry_widgets(left_frame)
        self._create_table_widgets(left_frame)
        self._create_visualization_widgets(right_frame)
        self._create_details_widgets(right_frame)
        self._create_menu()

    def _create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save State As...", command=self.save_state_dialog)
        file_menu.add_command(label="Load State...", command=self.load_state_dialog)
        
        # --- NEW: Edit Menu ---
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo (Ctrl+Z)", command=self.undo_action)
        edit_menu.add_command(label="Redo (Ctrl+Y)", command=self.redo_action)

        # Config Menu
        config_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Config", menu=config_menu)
        config_menu.add_command(label="Set C_puct...", command=self.set_c_puct_dialog)

    def _create_entry_widgets(self, parent):
        entry_frame = ttk.LabelFrame(parent, text="Add New Node", padding="10")
        entry_frame.pack(fill=tk.X, pady=5)

        # Input Fields
        ttk.Label(entry_frame, text="Parent ID:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.parent_id_var = tk.StringVar()
        self.parent_id_entry = ttk.Entry(entry_frame, textvariable=self.parent_id_var)
        self.parent_id_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)

        ttk.Label(entry_frame, text="ARI Score:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.ari_score_var = tk.StringVar()
        self.ari_score_entry = ttk.Entry(entry_frame, textvariable=self.ari_score_var)
        self.ari_score_entry.grid(row=1, column=1, sticky=tk.EW, padx=5)

        ttk.Label(entry_frame, text="Notes:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.notes_var = tk.StringVar()
        ttk.Entry(entry_frame, textvariable=self.notes_var).grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5)

        # Code Snippet Input
        ttk.Label(entry_frame, text="Code Snippet (Optional):").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.code_text = scrolledtext.ScrolledText(entry_frame, height=5)
        self.code_text.grid(row=4, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)

        # Add Button
        self.add_button = ttk.Button(entry_frame, text="Add Node", command=self.add_node_event)
        self.add_button.grid(row=5, column=1, columnspan=2, sticky=tk.E, pady=10)

        # Suggest Parent Button
        self.suggest_button = ttk.Button(entry_frame, text="Suggest Parent (Max PUCT)", command=self.suggest_parent)
        self.suggest_button.grid(row=0, column=2, sticky=tk.W, padx=5)

        entry_frame.columnconfigure(1, weight=1)

    def _create_table_widgets(self, parent):
        table_frame = ttk.LabelFrame(parent, text="Tree Data", padding="10")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        cols = ['ID', 'ParentID', 'ARI', 'V_u', 'RankScore', 'PUCT', 'Notes']
        self.treeview = ttk.Treeview(table_frame, columns=cols, show='headings')
        
        # Setup columns and headings with sorting capability
        for col in cols:
            self.treeview.heading(col, text=col, command=lambda c=col: self.sort_column(c, False))
            self.treeview.column(col, width=80, anchor=tk.CENTER)
        self.treeview.column('Notes', width=250, anchor=tk.W)

        # Scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.treeview.yview)
        self.treeview.configure(yscrollcommand=vsb.set)

        # Layout
        self.treeview.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Bind selection event to show details
        self.treeview.bind("<<TreeviewSelect>>", self.on_tree_select)

    def _create_visualization_widgets(self, parent):
        self.viz_frame = ttk.LabelFrame(parent, text="Tree Visualization", padding="10")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Matplotlib figure setup
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=85)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_details_widgets(self, parent):
        details_frame = ttk.LabelFrame(parent, text="Selected Node Details", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=False, pady=5)

        self.details_text = scrolledtext.ScrolledText(details_frame, height=12, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.details_text.config(state=tk.DISABLED) # Read-only

    # --- Event Handlers and Logic ---

    def _load_initial_state(self):
        # Automatically load the default state file if it exists
        if self.manager.load_state(self.state_file):
            self.refresh_ui()
            print(f"Loaded previous state from {self.state_file}")
        self.suggest_parent() # Suggest initial parent (0) if empty

    def save_state(self):
        # Auto-save functionality
        self.manager.save_state(self.state_file)

    def save_state_dialog(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.state_file = filepath
            self.save_state()
            messagebox.showinfo("Save State", f"State saved to {filepath}")

    def load_state_dialog(self):
        # Open a dialog to load a specific state file
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.state_file = filepath
            if self.manager.load_state(filepath):
                # --- MODIFIED: Clear history on new file load ---
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.refresh_ui()
                messagebox.showinfo("Load State", f"Successfully loaded state from {filepath}")
            else:
                messagebox.showerror("Error", "Failed to load state from file.")

    def set_c_puct_dialog(self):
        new_c_puct = simpledialog.askfloat("Set C_puct", f"Enter new C_puct value (Current: {self.manager.c_puct:.4f}):", parent=self)
        if new_c_puct is not None:
            # --- MODIFIED: Save state for undo before changing it ---
            self._save_state_for_undo()
            self.manager.set_c_puct(new_c_puct)
            self.refresh_ui()
            self.save_state() # Save config change

    def suggest_parent(self):
        # Automatically fill the Parent ID field with the node having the Max PUCT
        next_node = self.manager.get_next_node_to_explore()
        if next_node is not None:
            self.parent_id_var.set(str(int(next_node['ID'])))
        elif self.manager.df.empty:
            self.parent_id_var.set("0") # Suggest root (0) if tree is empty
        
    def add_node_event(self):
        try:
            # --- MODIFIED: Save state for undo before adding node ---
            self._save_state_for_undo()

            parent_id_str = self.parent_id_var.get()
            ari_score_str = self.ari_score_var.get()

            if not parent_id_str or not ari_score_str:
                messagebox.showerror("Error", "Parent ID and ARI Score are required.")
                self.undo_stack.pop() # Remove the saved state if action fails
                return

            parent_id = int(parent_id_str)
            ari_score = float(ari_score_str)
            notes = self.notes_var.get()
            code = self.code_text.get("1.0", tk.END).strip()

            # Basic validation
            if ari_score < -1 or ari_score > 1:
                 messagebox.showerror("Error", "ARI Score must be between -1 (error) and 1.")
                 self.undo_stack.pop() # Remove the saved state if action fails
                 return

            # Add the node and get its new ID
            node_id = self.manager.add_node(parent_id, ari_score, notes, code)
            
            # Create the corresponding script file
            self._create_node_script_file(node_id)

            self.refresh_ui()
            self.save_state() # Auto-save after adding a node
            self._clear_entry_fields()

        except ValueError as e:
            # Catches conversion errors and errors raised by manager logic
            self.undo_stack.pop() # Remove the saved state if action fails
            messagebox.showerror("Error", f"Invalid input or operation: {e}")
        except Exception as e:
            self.undo_stack.pop() # Remove the saved state if action fails
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def _create_node_script_file(self, node_id):
        """Copies sandbox.py to a new node-specific file in the /Nodes folder."""
        source_file = 'sandbox.py'
        dest_dir = 'Nodes'
        
        if not os.path.exists(source_file):
            # Print a warning to the console instead of a disruptive popup
            print(f"Warning: '{source_file}' not found. Skipping file creation for Node {node_id}.")
            return

        try:
            # Create the destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)

            # Define the destination filename (e.g., Node_001.py)
            dest_filename = f"Node_{node_id:03d}.py"
            dest_path = os.path.join(dest_dir, dest_filename)

            # Copy the file
            shutil.copy(source_file, dest_path)
            print(f"Successfully created node script: {dest_path}")

        except Exception as e:
            # Show a non-blocking warning if something goes wrong during file operation
            messagebox.showwarning("File Creation Warning", f"Could not create node script for Node {node_id}.\nError: {e}")

    # --- NEW: Undo/Redo Methods ---
    def _save_state_for_undo(self):
        """Saves a deep copy of the current state to the undo stack."""
        state = (self.manager.df.copy(), self.manager.c_puct)
        self.undo_stack.append(state)
        self.redo_stack.clear() # A new action invalidates the old redo history

    def _restore_state(self, state):
        """Restores the application to a given state."""
        df_to_restore, c_puct_to_restore = state
        self.manager.df = df_to_restore.copy()
        self.manager.c_puct = c_puct_to_restore
        
        # Recalculate everything based on the restored dataframe
        if not self.manager.df.empty:
            self.manager.next_id = self.manager.df['ID'].max() + 1
        else:
            self.manager.next_id = 1
        
        self.manager._calculate_metrics()
        self.refresh_ui()
        self.suggest_parent()

    def undo_action(self, event=None):
        """Pops from the undo stack and restores the state."""
        if not self.undo_stack:
            print("Undo stack is empty.")
            return

        # Save current state for redo before undoing
        current_state = (self.manager.df.copy(), self.manager.c_puct)
        self.redo_stack.append(current_state)

        state_to_restore = self.undo_stack.pop()
        self._restore_state(state_to_restore)
        print("Undo action performed.")

    def redo_action(self, event=None):
        """Pops from the redo stack and restores the state."""
        if not self.redo_stack:
            print("Redo stack is empty.")
            return
        
        # Save current state for undo before redoing
        current_state = (self.manager.df.copy(), self.manager.c_puct)
        self.undo_stack.append(current_state)

        state_to_restore = self.redo_stack.pop()
        self._restore_state(state_to_restore)
        print("Redo action performed.")

    def _clear_entry_fields(self):
        # Clear fields and suggest the next parent
        self.ari_score_var.set("")
        self.notes_var.set("")
        self.code_text.delete("1.0", tk.END)
        self.suggest_parent()
        self.ari_score_entry.focus_set() # Set focus for next input

    def refresh_ui(self):
        self._update_table()
        self._update_visualization()

    def _update_table(self):
        # Clear existing data
        for item in self.treeview.get_children():
            self.treeview.delete(item)
        
        # Insert new data from DataFrame
        df_display = self.manager.df.copy()
        # Format floats for display
        for col in ['ARI', 'RankScore', 'PUCT']:
             df_display[col] = df_display[col].map(lambda x: '{:.4f}'.format(x) if not pd.isna(x) else 'NaN')

        for index, row in df_display.iterrows():
            # Ensure IDs and V_u are displayed as integers
            values = [int(row['ID']), int(row['ParentID']), row['ARI'], int(row['V_u']), row['RankScore'], row['PUCT'], row['Notes']]
            self.treeview.insert("", tk.END, values=values)

    def _update_visualization(self):
        self.ax.clear()
        G = self.manager.build_graph()
        if G.number_of_nodes() == 0:
            self.canvas.draw()
            return

        # Layout: Kamada-Kawai often provides a readable visualization for graphs
        try:
            # For larger graphs, spring_layout can be faster
            if G.number_of_nodes() > 50:
                pos = nx.spring_layout(G, seed=42)
            else:
                pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42) # Fallback layout

        # Determine colors based on ARI score (using Viridis colormap)
        ari_scores = [G.nodes[n].get('ARI', 0) for n in G.nodes()]
        
        # Get highlights
        best_node = self.manager.get_best_node()
        best_node_id = int(best_node['ID']) if best_node is not None else None
        next_node = self.manager.get_next_node_to_explore()
        next_node_id = int(next_node['ID']) if next_node is not None else None

        # Draw the base graph
        nx.draw_networkx(G, pos, ax=self.ax, 
                         node_color=ari_scores, cmap=plt.cm.viridis, 
                         node_size=500, 
                         arrowstyle='->', arrowsize=10,
                         font_size=9, font_color='white', font_weight='bold')

        # Apply highlights (drawing specific nodes again with edge colors)
        # Highlight Max ARI (Green Border)
        if best_node_id:
             node_color_val = [G.nodes[best_node_id]['ARI']]
             nx.draw_networkx_nodes(G, pos, nodelist=[best_node_id], node_size=500, 
                                    edgecolors='green', linewidths=3, 
                                    node_color=node_color_val, cmap=plt.cm.viridis, vmin=min(ari_scores), vmax=max(ari_scores))
        
        # Highlight Max PUCT (Red Border), unless it's the same as Max ARI
        if next_node_id and next_node_id != best_node_id:
             node_color_val = [G.nodes[next_node_id]['ARI']]
             nx.draw_networkx_nodes(G, pos, nodelist=[next_node_id], node_size=500, 
                                    edgecolors='red', linewidths=3, 
                                    node_color=node_color_val, cmap=plt.cm.viridis, vmin=min(ari_scores), vmax=max(ari_scores))

        self.ax.set_title(f"|T|={len(self.manager.df)}, N_total={self.manager.N_total}, C_puct={self.manager.c_puct:.3f}\nGreen=Max ARI, Red=Max PUCT")
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

    def on_tree_select(self, event):
        # Display details of the selected node in the details panel
        selected_items = self.treeview.selection()
        if not selected_items:
            return
        
        item = self.treeview.item(selected_items[0])
        try:
            node_id = int(item['values'][0])
        except (ValueError, IndexError):
            return
        
        # Find the corresponding row in the manager DataFrame
        node_data = self.manager.df[self.manager.df['ID'] == node_id].iloc[0]
        
        # Update the details panel
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert(tk.END, f"ID: {int(node_data['ID'])}\n")
        self.details_text.insert(tk.END, f"Parent: {int(node_data['ParentID'])}\n")
        self.details_text.insert(tk.END, f"ARI: {node_data['ARI']:.4f}\n")
        self.details_text.insert(tk.END, f"PUCT: {node_data['PUCT']:.4f}\n")
        self.details_text.insert(tk.END, f"Visits (V_u): {int(node_data['V_u'])}\n")
        self.details_text.insert(tk.END, f"Notes: {node_data['Notes']}\n\n")
        self.details_text.insert(tk.END, f"--- Code Snippet ---\n{node_data['Code']}")
        self.details_text.config(state=tk.DISABLED)

    def sort_column(self, col, reverse):
        # Helper function for sorting the Treeview columns
        l = [(self.treeview.set(k, col), k) for k in self.treeview.get_children('')]
        
        # Try sorting numerically if possible, otherwise sort alphabetically
        def try_convert(val):
            try:
                return float(val)
            except ValueError:
                return val

        l.sort(key=lambda t: try_convert(t[0]), reverse=reverse)

        # Rearrange items in sorted positions
        for index, (val, k) in enumerate(l):
            self.treeview.move(k, '', index)

        # Reverse sort direction for the next click
        self.treeview.heading(col, command=lambda: self.sort_column(col, not reverse))

# =====================================================================
# Main execution block
# =====================================================================

if __name__ == "__main__":
    # Fix DPI awareness for Windows (improves clarity on high-res screens)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = PUCTTrackerApp()
    app.mainloop()