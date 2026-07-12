import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import ttk, filedialog, messagebox
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dateutil.relativedelta import relativedelta
import os
import pickle
import calendar
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

class IncidentPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Incident Volume Predictor")
        self.root.geometry("1200x800")
        self.model_path = "incident_models.pkl"
        self.data = None
        self.models = {}
        self.label_encoders = {}
        self.setup_ui()
        
        try:
            with open(self.model_path, "rb") as f:
                saved_data = pickle.load(f)
                if isinstance(saved_data, dict):
                    self.models = saved_data.get('models', {})
                    self.label_encoders = saved_data.get('label_encoders', {})
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            self.models = {}
            self.label_encoders = {}

    def setup_ui(self):
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        upload_frame = LabelFrame(main_frame, text="Data Input", padx=5, pady=5)
        upload_frame.pack(fill=X, pady=(0, 10))

        Label(upload_frame, text="Excel File:").grid(row=0, column=0, padx=5, pady=5, sticky=E)
        self.file_entry = Entry(upload_frame, width=50)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5)
        Button(upload_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)

        Label(upload_frame, text="Months to Predict:").grid(row=1, column=0, padx=5, pady=5, sticky=E)
        self.months_entry = Entry(upload_frame, width=10)
        self.months_entry.grid(row=1, column=1, padx=5, pady=5, sticky=W)
        self.months_entry.insert(0, "6")

        Button(upload_frame, text="Process & Predict", command=self.process_data).grid(row=1, column=2, padx=5, pady=5)

        self.viz_frame = LabelFrame(main_frame, text="Predictions Visualization", padx=5, pady=5)
        self.viz_frame.pack(fill=BOTH, expand=True)

        self.notebook = ttk.Notebook(self.viz_frame)
        self.notebook.pack(fill=BOTH, expand=True)

        self.tab_assignment_group = Frame(self.notebook)
        self.tab_priority = Frame(self.notebook)
        self.tab_category_combo = Frame(self.notebook)
        self.tab_spike_days = Frame(self.notebook)

        self.notebook.add(self.tab_assignment_group, text="Assignment Group Volumes")
        self.notebook.add(self.tab_priority, text="Priority Volumes")
        self.notebook.add(self.tab_category_combo, text="Category-SubCategory Volumes")
        self.notebook.add(self.tab_spike_days, text="Spike Days Prediction")

        self.status_var = StringVar()
        self.status_var.set("Ready")
        status_bar = Label(self.root, textvariable=self.status_var, bd=1, relief=SUNKEN, anchor=W)
        status_bar.pack(side=BOTTOM, fill=X)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if filename:
            self.file_entry.delete(0, END)
            self.file_entry.insert(0, filename)

    def process_data(self):
        file_path = self.file_entry.get()
        if not file_path:
            messagebox.showerror("Error", "Please select an Excel file")
            return

        try:
            months_to_predict = int(self.months_entry.get())
            if months_to_predict <= 0:
                raise ValueError("Months must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of months (positive integer)")
            return

        self.status_var.set("Processing data...")
        self.root.update_idletasks()

        try:
            df = pd.read_excel(file_path)
            
            required_cols = {"Assignment Group", "Priority", "Created", "Category", "Sub Category"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                messagebox.showerror("Error", f"Missing required columns: {', '.join(missing_cols)}")
                return

            df = df[list(required_cols)].copy()
            df['Created'] = pd.to_datetime(df['Created']).dt.date
            df.dropna(inplace=True)

            if df.empty:
                messagebox.showerror("Error", "No valid data found after cleaning")
                return

            self.data = df
            self.prepare_models()
            predictions = self.generate_predictions(months_to_predict)
            self.visualize_predictions(predictions, months_to_predict)
            
            self.status_var.set("Predictions complete. Ready")
            self.save_models()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred")

    def prepare_models(self):
        if self.data is None:
            return

        min_date = pd.to_datetime(self.data['Created'].min())
        max_date = pd.to_datetime(self.data['Created'].max())
        all_dates = pd.date_range(min_date, max_date, freq='D').date

        # Assignment Group models
        ag_daily = self.data.groupby(['Created', 'Assignment Group']).size().unstack(fill_value=0)
        ag_daily.index = pd.to_datetime(ag_daily.index)
        ag_daily = ag_daily.reindex(all_dates, fill_value=0)
        self.train_time_series_models(ag_daily, "assignment_group")

        # Priority models
        priority_daily = self.data.groupby(['Created', 'Priority']).size().unstack(fill_value=0)
        priority_daily = priority_daily.reindex(all_dates, fill_value=0)
        self.train_time_series_models(priority_daily, "priority")

        # Category-SubCategory models
        self.data['Category_SubCategory'] = self.data['Category'] + " - " + self.data['Sub Category']
        cat_daily = self.data.groupby(['Created', 'Category_SubCategory']).size().unstack(fill_value=0)
        cat_daily = cat_daily.reindex(all_dates, fill_value=0)
        self.train_time_series_models(cat_daily, "category_subcategory")

        # Complex models (monthly)
        monthly_data = self.data.copy()
        monthly_data['YearMonth'] = monthly_data['Created'].apply(lambda x: x.strftime('%Y-%m'))
        complex_model_data = monthly_data.groupby(
            ['YearMonth', 'Assignment Group', 'Category_SubCategory', 'Priority']
        ).size().reset_index(name='Count')
        complex_model_data = complex_model_data.sort_values('YearMonth')
        for ag in monthly_data['Assignment Group'].unique():
            ag_data = complex_model_data[complex_model_data['Assignment Group'] == ag]
            if len(ag_data) < 10:
                continue
                
            X = ag_data[['Category_SubCategory', 'Priority']].copy()
            y = ag_data['Count']
            
            for col in ['Category_SubCategory', 'Priority']:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(X[col])
                
                X[col] = self.label_encoders[col].transform(X[col])
            
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            if not hasattr(self, 'eval_metrics'):
                self.eval_metrics = {}
            #self.eval_metrics[f"{prefix}_{col}"] = {'mae': mae, 'r2': r2}
            #print(f"{prefix}_{col} — R²: {r2:.3f}, MAE: {mae:.2f}")
            self.eval_metrics[f"complex_{ag}"] = {'mae': mae, 'r2': r2}
            print(f"complex_{ag} — R²: {r2:.3f}, MAE: {mae:.2f}")
            self.models[f"complex_{ag}"] = model

    def train_time_series_models(self, df, prefix):
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()    
        df['days_since_start'] = (df.index - pd.Timestamp(df.index[0])).days
        #df['days_since_start'] = (pd.to_datetime(df.index) - pd.to_datetime(df.index[0])).dt.days
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year

        for col in df.columns:
            if col in ['days_since_start', 'day_of_week', 'day_of_month', 'month', 'year']:
                continue
                
            y = df[col]
            X = df[['days_since_start', 'day_of_week', 'day_of_month', 'month', 'year']]
            
            if y.sum() == 0:
                continue
                
            if len(y) < 30:
                continue
                
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            
            
            y_pred_test = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            if not hasattr(self, 'eval_metrics'):
                self.eval_metrics = {}
            self.eval_metrics[f"{prefix}_{col}"] = {'mae': mae, 'r2': r2}
            print(f"{prefix}_{col} — R²: {r2:.3f}, MAE: {mae:.2f}")
           
            
            
            self.models[f"{prefix}_{col}"] = model

    def generate_predictions(self, months_to_predict):
        if self.data is None:
            return None

        today = datetime.now().date()
        future_dates = [today + timedelta(days=i) for i in range(months_to_predict * 30)]
        #future_df = pd.DataFrame(index=future_dates)
        min_date = self.data['Created'].min()
        future_df = pd.DataFrame(index=future_dates)
        future_df['days_since_start'] = (pd.to_datetime(future_df.index) - pd.to_datetime(min_date)).days
        future_df['day_of_week'] = [d.weekday() for d in future_df.index]
        future_df['day_of_month'] = [d.day for d in future_df.index]
        future_df['month'] = [d.month for d in future_df.index]
        future_df['year'] = [d.year for d in future_df.index]

        predictions = {
            'assignment_group': {},
            'priority': {},
            'category_subcategory': {},
            'spike_days': defaultdict(list),
            'complex': defaultdict(dict)
        }

        # Predict Assignment Group volumes
        for col in self.data['Assignment Group'].unique():
            model_key = f"assignment_group_{col}"
            if model_key in self.models:
                X = future_df[['days_since_start', 'day_of_week', 'day_of_month', 'month', 'year']]
                preds = self.models[model_key].predict(X)
                predictions['assignment_group'][col] = preds
                
                # Only consider spikes if we have predictions
                if len(preds) > 0:
                    threshold = np.percentile(preds, 90)
                    spike_days = future_df.index[preds >= threshold]
                    for day in spike_days:
                        predictions['spike_days'][day].append((col, preds[future_df.index.get_loc(day)]))

        # Predict Priority volumes
        for col in self.data['Priority'].unique():
            model_key = f"priority_{col}"
            if model_key in self.models:
                X = future_df[['days_since_start', 'day_of_week', 'day_of_month', 'month', 'year']]
                preds = self.models[model_key].predict(X)
                predictions['priority'][col] = preds

        # Predict Category-SubCategory volumes
        for col in self.data['Category_SubCategory'].unique():
            model_key = f"category_subcategory_{col}"
            if model_key in self.models:
                X = future_df[['days_since_start', 'day_of_week', 'day_of_month', 'month', 'year']]
                preds = self.models[model_key].predict(X)
                predictions['category_subcategory'][col] = preds

        # Predict complex relationships
        for ag in self.data['Assignment Group'].unique():
            model_key = f"complex_{ag}"
            if model_key in self.models:
                model = self.models[model_key]
                
                category_combos = self.data[self.data['Assignment Group'] == ag]['Category_SubCategory'].unique()
                priorities = self.data[self.data['Assignment Group'] == ag]['Priority'].unique()
                
                for month_offset in range(months_to_predict):
                    month_start = today + relativedelta(months=month_offset)
                    month_name = month_start.strftime('%Y-%m')
                    
                    for cat_combo in category_combos:
                        for priority in priorities:
                            input_data = pd.DataFrame({
                                'Category_SubCategory': [cat_combo],
                                'Priority': [priority]
                            })
                            
                            for col in ['Category_SubCategory', 'Priority']:
                                if col in self.label_encoders:
                                    input_data[col] = self.label_encoders[col].transform(input_data[col])
                            
                            pred_count = model.predict(input_data)[0]
                            predictions['complex'][ag][(month_name, cat_combo, priority)] = max(0, pred_count)

        return predictions
    """
    def visualize_predictions(self, predictions, months_to_predict):
        for widget in self.tab_assignment_group.winfo_children():
            widget.destroy()
        for widget in self.tab_priority.winfo_children():
            widget.destroy()
        for widget in self.tab_category_combo.winfo_children():
            widget.destroy()
        for widget in self.tab_spike_days.winfo_children():
            widget.destroy()

        # Assignment Group visualization
        if predictions['assignment_group']:
            #fig, ax = plt.subplots(figsize=(10, 6))
            fig, ax = plt.subplots(figsize=(10, months_to_predict))
            for ag, values in predictions['assignment_group'].items():
                monthly_volumes = []
                for month_offset in range(months_to_predict):
                    start_idx = month_offset * 30
                    end_idx = min((month_offset + 1) * 30, len(values))
                    monthly_volumes.append(np.sum(values[start_idx:end_idx]))
                
                months = [datetime.now().replace(day=1) + relativedelta(months=i) for i in range(months_to_predict)]
                month_names = [m.strftime('%b %Y') for m in months]
                ax.bar(month_names, monthly_volumes, label=ag, alpha=0.7)
            
            ax.set_title('Predicted Monthly Incident Volumes by Assignment Group')
            ax.set_ylabel('Number of Incidents')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=self.tab_assignment_group)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        # Priority visualization
        if predictions['priority']:
            #fig, ax = plt.subplots(figsize=(10, 6))
            fig, ax = plt.subplots(figsize=(20, months_to_predict))
            priority_data = {p: np.sum(values) for p, values in predictions['priority'].items()}
            
            wedges, texts, autotexts = ax.pie(
                priority_data.values(), 
                labels=priority_data.keys(), 
                autopct='%1.1f%%',
                startangle=90
            )
            ax.set_title('Predicted Incident Distribution by Priority')
            
            canvas = FigureCanvasTkAgg(fig, master=self.tab_priority)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        # Category-SubCategory visualization
        if predictions['category_subcategory']:
            total_by_category = {cat: np.sum(values) for cat, values in predictions['category_subcategory'].items()}
            top_categories = sorted(total_by_category.items(), key=lambda x: x[1], reverse=True)[:10]
            
            #fig, ax = plt.subplots(figsize=(10, 6))
            fig, ax = plt.subplots(figsize=(10, months_to_predict))
            y_pos = np.arange(len(top_categories))
            ax.barh(
                y_pos,
                [vol for _, vol in top_categories],
                align='center'
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels([cat for cat, _ in top_categories])
            ax.invert_yaxis()
            ax.set_title('Top Predicted Incident Categories (Volume)')
            ax.set_xlabel('Number of Incidents')
            
            canvas = FigureCanvasTkAgg(fig, master=self.tab_category_combo)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        # Spike Days visualization
        if predictions['spike_days']:
            spike_days = predictions['spike_days']
            spike_list = [(day, ag_list) for day, ag_list in spike_days.items()]
            spike_list.sort(key=lambda x: x[0])
            spike_list = spike_list[-20:] if len(spike_list) > 20 else spike_list
            
            #fig, ax = plt.subplots(figsize=(12, 6))
            fig, ax = plt.subplots(figsize=(12, months_to_predict))
            y_pos = np.arange(len(spike_list))
            
            for i, (day, ag_list) in enumerate(spike_list):
                total = sum(vol for _, vol in ag_list)
                ax.barh(y_pos[i], total, label=day.strftime('%Y-%m-%d'))
                
                bottom = 0
                for ag, vol in ag_list:
                    ax.barh(y_pos[i], vol, left=bottom, label=f"{ag}: {int(vol)}")
                    bottom += vol
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([day.strftime('%Y-%m-%d') for day, _ in spike_list])
            ax.set_xlabel('Incident Volume')
            ax.set_title('Predicted Spike Days with Assignment Group Breakdown')
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=self.tab_spike_days)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        # Complex visualization
        if predictions['complex']:
            for ag in predictions['complex']:
                if not predictions['complex'][ag]:
                    continue
                
                ag_frame = Frame(self.notebook)
                self.notebook.add(ag_frame, text=f"{ag[:15]}... Details" if len(ag) > 15 else f"{ag} Details")
                
                ag_data = predictions['complex'][ag]
                months = sorted(set(month for month, _, _ in ag_data.keys()))
                categories = sorted(set(cat for _, cat, _ in ag_data.keys()))
                priorities = sorted(set(priority for _, _, priority in ag_data.keys()))
                
                #fig, axes = plt.subplots(len(months), 1, figsize=(12, 6*len(months)))
                fig, axes = plt.subplots(len(months), 1, figsize=(12, months_to_predict*len(months)))
                if len(months) == 1:
                    axes = [axes]
                
                for i, month in enumerate(months):
                    bottom = None
                    for priority in priorities:
                        values = [ag_data.get((month, cat, priority), 0) for cat in categories]
                        label = str(priority)
                        
                        if bottom is None:
                            axes[i].bar(categories, values, label=label)
                            bottom = values
                        else:
                            axes[i].bar(categories, values, bottom=bottom, label=label)
                            bottom = [b + v for b, v in zip(bottom, values)]
                    
                    axes[i].set_title(f"{month}")
                    axes[i].legend(title="Priority")
                    axes[i].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=ag_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=BOTH, expand=True)
"""

    def visualize_predictions(self, predictions, months_to_predict):
        # Clear existing widgets
        for widget in self.tab_assignment_group.winfo_children():
            widget.destroy()
        for widget in self.tab_priority.winfo_children():
            widget.destroy()
        for widget in self.tab_category_combo.winfo_children():
            widget.destroy()
        for widget in self.tab_spike_days.winfo_children():
            widget.destroy()

        # Common style settings
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 8

        # Color palette
        color_palette = plt.cm.tab20.colors
        bar_colors = color_palette[:10]  # Use first 10 colors from tab20 palette

        # Helper function to create scrollable frame
        def create_scrollable_frame(parent):
            container = Frame(parent)
            container.pack(fill=BOTH, expand=True)
            
            canvas = Canvas(container)
            canvas.pack(side=LEFT, fill=BOTH, expand=True)
            
            scrollbar = Scrollbar(container, orient=VERTICAL, command=canvas.yview)
            scrollbar.pack(side=RIGHT, fill=Y)
            canvas.configure(yscrollcommand=scrollbar.set)
            
            scrollable_frame = Frame(canvas)
            canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
            
            def on_frame_configure(event):
                canvas.configure(scrollregion=canvas.bbox("all"))
            scrollable_frame.bind("<Configure>", on_frame_configure)
            
            def on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", on_mousewheel)
            
            return scrollable_frame

        # Assignment Group visualization
        if predictions.get('assignment_group'):
            scrollable_frame = create_scrollable_frame(self.tab_assignment_group)
            
            fig, ax = plt.subplots(figsize=(10, max(6, months_to_predict * 0.8)))
            fig.patch.set_facecolor('#f5f5f5')
            
            # Calculate monthly volumes
            monthly_data = {}
            for ag, values in predictions['assignment_group'].items():
                monthly_volumes = []
                for month_offset in range(months_to_predict):
                    start_idx = month_offset * 30
                    end_idx = min((month_offset + 1) * 30, len(values))
                    monthly_volumes.append(np.sum(values[start_idx:end_idx]))
                monthly_data[ag] = monthly_volumes
            
            # Get month labels
            months = [datetime.now().replace(day=1) + relativedelta(months=i) for i in range(months_to_predict)]
            month_names = [m.strftime('%b %Y') for m in months]
            
            # Plot stacked bar chart
            bottom = None
            color_idx = 0
            for ag, volumes in monthly_data.items():
                if bottom is None:
                    ax.bar(month_names, volumes, color=bar_colors[color_idx % len(bar_colors)], 
                        label=ag, alpha=0.8, edgecolor='white')
                    bottom = volumes
                else:
                    ax.bar(month_names, volumes, bottom=bottom, color=bar_colors[color_idx % len(bar_colors)], 
                        label=ag, alpha=0.8, edgecolor='white')
                    bottom = [b + v for b, v in zip(bottom, volumes)]
                color_idx += 1
            
            ax.set_title('Predicted Monthly Incident Volumes by Assignment Group', pad=20)
            ax.set_ylabel('Number of Incidents')
            ax.set_xlabel('Month')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Adjust legend
            if monthly_data:  # Only create legend if there's data
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], 
                        bbox_to_anchor=(1.05, 1), 
                        loc='upper left', 
                        borderaxespad=0.,
                        title="Assignment Groups")
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Priority visualization
        if predictions.get('priority'):
            scrollable_frame = create_scrollable_frame(self.tab_priority)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('#f5f5f5')
            
            priority_data = {p: np.sum(values) for p, values in predictions['priority'].items()}
            
            # Sort priorities for consistent order
            sorted_priorities = sorted(priority_data.items(), key=lambda x: x[0])
            priorities = [p[0] for p in sorted_priorities]
            volumes = [p[1] for p in sorted_priorities]
            
            if sum(volumes) > 0:  # Only plot if there are values
                # Create pie chart with labels outside
                wedges, texts, autotexts = ax.pie(
                    volumes,
                    labels=priorities,
                    colors=bar_colors[:len(priorities)],
                    autopct=lambda p: f'{p:.1f}%\n({int(p*sum(volumes)/100)})',
                    startangle=90,
                    pctdistance=0.85,
                    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                    textprops={'color': 'black', 'fontsize': 10},
                    labeldistance=1.1
                )
                
                # Adjust text positions for better visibility
                for text in texts + autotexts:
                    text.set(size=10)
                    
                # Equal aspect ratio ensures pie is drawn as a circle
                ax.axis('equal')
                ax.set_title('Predicted Incident Distribution by Priority', pad=20)
                
                # Add legend outside
                ax.legend(wedges, priorities,
                        title="Priorities",
                        loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1))
                
                plt.tight_layout()
                
                canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Category-SubCategory visualization
        if predictions.get('category_subcategory'):
            scrollable_frame = create_scrollable_frame(self.tab_category_combo)
            
            total_by_category = {cat: np.sum(values) for cat, values in predictions['category_subcategory'].items()}
            top_categories = [item for item in sorted(total_by_category.items(), key=lambda x: x[1], reverse=True) if item[1] > 0][:10]
            
            if top_categories:  # Only plot if there are categories with values
                fig, ax = plt.subplots(figsize=(10, 8 + len(top_categories)*0.1))  # Dynamic height
                fig.patch.set_facecolor('#f5f5f5')
                
                y_pos = np.arange(len(top_categories))
                bars = ax.barh(
                    y_pos,
                    [vol for _, vol in top_categories],
                    color=bar_colors[:len(top_categories)],
                    align='center',
                    alpha=0.8,
                    edgecolor='white'
                )
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels([cat[:50] + '...' if len(cat) > 50 else cat for cat, _ in top_categories])
                ax.invert_yaxis()
                ax.set_title('Top Predicted Incident Categories (Volume)', pad=20)
                ax.set_xlabel('Number of Incidents')
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Add value labels
                max_val = max([v for _, v in top_categories]) if top_categories else 0
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + max_val * 0.01,
                            bar.get_y() + bar.get_height()/2,
                            f'{int(width)}',
                            ha='left', va='center', fontsize=9)
                
                plt.tight_layout()
                
                canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Spike Days visualization
        if predictions.get('spike_days'):
            scrollable_frame = create_scrollable_frame(self.tab_spike_days)
            
            spike_days = predictions['spike_days']
            spike_list = [(day, ag_list) for day, ag_list in spike_days.items() if any(v > 0 for _, v in ag_list)]
            spike_list.sort(key=lambda x: x[0])
            spike_list = spike_list[-20:] if len(spike_list) > 20 else spike_list
            
            if spike_list:  # Only plot if there are spike days
                fig_height = max(6, len(spike_list) * 0.6)
                fig, ax = plt.subplots(figsize=(12, fig_height))
                fig.patch.set_facecolor('#f5f5f5')
                
                height = 0.8
                y_pos = np.arange(len(spike_list)) * (height + 0.2)
                
                legend_handles = []
                legend_labels = []
                
                for i, (day, ag_list) in enumerate(spike_list):
                    bottom = 0
                    for ag, vol in sorted(ag_list, key=lambda x: x[1], reverse=True):
                        if vol > 0:  # Only plot if volume > 0
                            if ag not in legend_labels:
                                bar = ax.barh(y_pos[i], vol, left=bottom, 
                                            color=bar_colors[len(legend_handles) % len(bar_colors)], 
                                            height=height, label=ag)
                                legend_handles.append(bar)
                                legend_labels.append(ag)
                            else:
                                idx = legend_labels.index(ag)
                                ax.barh(y_pos[i], vol, left=bottom, 
                                    color=bar_colors[idx % len(bar_colors)], 
                                    height=height)
                            bottom += vol
                
                if spike_list:  # Only set labels if there are items
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([day.strftime('%Y-%m-%d') for day, _ in spike_list])
                    ax.set_xlabel('Incident Volume')
                    ax.set_title('Predicted Spike Days with Assignment Group Breakdown', pad=20)
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    
                    # Create legend if there are any handles
                    if legend_handles:
                        ax.legend(legend_handles, legend_labels, 
                                title="Assignment Groups",
                                bbox_to_anchor=(1.05, 1), 
                                loc='upper left')
                    
                    plt.tight_layout()
                    
                    canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)
        """
        # Complex visualization
        if predictions.get('complex'):
            for ag in predictions['complex']:
                if not predictions['complex'][ag]:
                    continue
                    
                ag_frame = Frame(self.notebook)
                self.notebook.add(ag_frame, text=f"{ag[:15]}... Details" if len(ag) > 15 else f"{ag} Details")
                scrollable_frame = create_scrollable_frame(ag_frame)
                
                ag_data = predictions['complex'][ag]
                months = sorted(set(month for month, _, _ in ag_data.keys()))
                categories = sorted(set(cat for _, cat, _ in ag_data.keys()))
                priorities = sorted(set(priority for _, _, priority in ag_data.keys()))
                
                fig, axes = plt.subplots(len(months), 1, figsize=(12, max(6, len(months) * 4)))
                fig.patch.set_facecolor('#f5f5f5')
                
                if len(months) == 1:
                    axes = [axes]
                
                for i, month in enumerate(months):
                    bottom = None
                    legend_handles = []
                    legend_labels = []
                    
                    for priority_idx, priority in enumerate(priorities):
                        values = [ag_data.get((month, cat, priority), 0) for cat in categories]
                        if sum(values) == 0:
                            continue
                        
                        label = str(priority)
                        
                        if bottom is None:
                            bars = axes[i].bar(categories, values, 
                                            color=bar_colors[priority_idx % len(bar_colors)], 
                                            label=label, alpha=0.8)
                            bottom = values
                        else:
                            bars = axes[i].bar(categories, values, bottom=bottom, 
                                            color=bar_colors[priority_idx % len(bar_colors)], 
                                            label=label, alpha=0.8)
                            bottom = [b + v for b, v in zip(bottom, values)]
                        
                        legend_handles.append(bars[0])
                        legend_labels.append(label)
                    
                    axes[i].set_title(f"{month}", pad=10)
                    if legend_handles:  # Only add legend if there are handles
                        axes[i].legend(legend_handles, legend_labels, title="Priority")
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Add value labels for the top of each stack
                    if bottom:  # Only add labels if there's data
                        for cat_idx, cat in enumerate(categories):
                            total = sum(ag_data.get((month, cat, priority), 0) for priority in priorities)
                            if total > 0:
                                axes[i].text(cat_idx, bottom[cat_idx], f'{int(total)}',
                                        ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=10, pady=10)
                """

    def save_models(self):
        with open(self.model_path, "wb") as f:
            pickle.dump({
                'models': self.models,
                'label_encoders': self.label_encoders
            }, f)

def main():
    root = Tk()
    app = IncidentPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
