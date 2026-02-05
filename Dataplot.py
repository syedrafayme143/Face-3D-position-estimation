import dearpygui.dearpygui as dpg
import os.path

class DataPlot:
    """
    Real-time data plotting widget with optional recording capabilities.
    """
    
    def __init__(self, labels, maxnumpoints, showRecordPause=False):
        """
        Initialize the DataPlot.
        
        Args:
            labels: Tuple/list of curve labels
            maxnumpoints: Maximum number of points to display before scrolling
            showRecordPause: Whether to show record/pause controls
        """
        # Initialize data storage for each curve
        self.curves_ = [[] for _ in range(len(labels))]
        self.timedata_ = []
        self.series_ = []
        self.labels_ = labels
        self.maxnumpoints_ = maxnumpoints
        
        # Recording state
        self.recordFile_ = None
        self.fileFirstLine_ = False
        
        # Pause state
        self.paused_ = False
        self.showRecordPause_ = showRecordPause

    def recordCallback(self):
        """Toggle recording on/off."""
        if self.recordFile_ is None:
            # Start recording
            fname = dpg.get_value(self.csvText_)
            if fname == "":
                fname = "bb8.csv"
            elif not fname.endswith(".csv"):
                fname = fname + ".csv"
            
            # Check if file exists
            if os.path.isfile(fname):
                print(f"Appending to {fname}")
                self.recordFile_ = open(fname, "a")
                self.fileFirstLine_ = False
            else:
                print(f"Creating {fname}")
                self.recordFile_ = open(fname, "w")
                self.fileFirstLine_ = True
            
            dpg.configure_item(self.recordButton_, label="Stop")
        else:
            # Stop recording
            self.recordFile_.close()
            self.recordFile_ = None
            dpg.configure_item(self.recordButton_, label="Record")

    def pauseCallback(self):
        """Toggle pause on/off."""
        if self.paused_:
            dpg.configure_item(self.pauseButton_, label="Pause")
            self.paused_ = False
        else:
            dpg.configure_item(self.pauseButton_, label="Continue")
            self.paused_ = True

    def createGUI(self, width, height, label=""):
        """
        Create the GUI elements for the plot.
        
        Args:
            width: Plot width (-1 for auto)
            height: Plot height (-1 for auto)
            label: Plot title
        """
        # Optional record/pause controls
        if self.showRecordPause_:
            with dpg.group(horizontal=True):
                self.recordButton_ = dpg.add_button(
                    label="Record", 
                    callback=self.recordCallback
                )
                self.csvText_ = dpg.add_input_text(
                    label="CSV", 
                    default_value="bb8.csv"
                )
                self.pauseButton_ = dpg.add_button(
                    label="Pause", 
                    callback=self.pauseCallback
                )

        # Create plot
        with dpg.plot(label=label, width=width, height=height):
            dpg.add_plot_legend()
            self.xAxis_ = dpg.add_plot_axis(dpg.mvXAxis, label="Frame")
            self.yAxis_ = dpg.add_plot_axis(dpg.mvYAxis, label="Position (mm)")
            
            # Create line series for each curve
            for i in range(len(self.labels_)):
                self.series_.append(
                    dpg.add_line_series(
                        [], [], 
                        label=self.labels_[i], 
                        parent=self.yAxis_
                    )
                )

    def addDataVector(self, timestamp, vector):
        """
        Add a new data point to all curves.
        
        Args:
            timestamp: Time or frame number
            vector: Tuple/list of values (one per curve)
        """
        # Validate input
        if len(vector) != len(self.labels_):
            print(
                f"Error: Trying to add {len(vector)}-vector to data plot "
                f"with {len(self.labels_)} curves"
            )
            return

        # Add data points and maintain max size
        for i in range(len(vector)):
            self.curves_[i].append(vector[i])
            if len(self.curves_[i]) > self.maxnumpoints_:
                self.curves_[i].pop(0)
        
        self.timedata_.append(timestamp)
        if len(self.timedata_) > self.maxnumpoints_:
            self.timedata_.pop(0)

        # Update plot if not paused
        if not self.paused_:
            self._updatePlot()

        # Record data if recording is active
        if self.recordFile_ is not None:
            self._recordData(timestamp, vector)

    def _updatePlot(self):
        """Update the plot with current data."""
        # Update each series
        for i in range(len(self.curves_)):
            dpg.set_value(self.series_[i], [self.timedata_, self.curves_[i]])
        
        # Update x-axis limits
        if len(self.timedata_) > 0:
            dpg.set_axis_limits(self.xAxis_, self.timedata_[0], self.timedata_[-1])
        
        # Auto-fit on first entry
        if len(self.timedata_) == 1:
            self.autofit()

    def _recordData(self, timestamp, vector):
        """
        Record data to CSV file.
        
        Args:
            timestamp: Time or frame number
            vector: Data values
        """
        # Write header on first line
        if self.fileFirstLine_:
            header = "# timestamp; " + "; ".join(self.labels_)
            print(header, file=self.recordFile_)
            self.fileFirstLine_ = False
        
        # Write data row
        data_str = f"{timestamp:.6f}; " + "; ".join(f"{v:.6f}" for v in vector)
        print(data_str, file=self.recordFile_)
        self.recordFile_.flush()  # Ensure data is written immediately

    def autofit(self):
        """Auto-fit the y-axis to show all data."""
        dpg.fit_axis_data(self.yAxis_)