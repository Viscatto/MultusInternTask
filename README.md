This project provides a modular framework for automated **cell counting of Mesenchymal Stem Cells (MSCs) from brightfield microscopy images**.

The pipeline includes both classical computer vision techniques as well as advanced deep-learning integration, designed to handle the challenges of low-contrast, 
irregularly shaped cell morphologies.

**Core Components**
main.py: The primary execution script. It serves as the user interface for selecting counting methodologies, configuring hyperparameters, and running the processing pipeline.

base_cell_counter.py: The foundational parent class. It centralizes shared logic for image loading, data annotation, and result preservation, ensuring that all specific counting models (children) maintain a consistent architecture.

summary_report.py: A dedicated reporting module that processes numerical outputs to generate summary statistics and visualizations, such as cell-count fluctuations over time.
