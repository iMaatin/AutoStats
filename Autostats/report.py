### imports 
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter,A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from datetime import datetime
from reportlab.platypus import Table, TableStyle
from scipy.stats import shapiro, probplot 
import os 
import textwrap

auto_version = '0.1.3'

def auto_report(df: pd.DataFrame, tresh: int = 10, output_file: str = "report.pdf", df_name: str = "Dataset") -> None:
    """
    Generate a PDF report for data exploration and analysis resized for A4 paper.

    The report includes:
      - A cover page with an ASCII art logo, metadata (DataFrame name, analysis date/time),
        and library version information.
      - Column categorization based on data type and uniqueness:
          * Categorical columns: non-numeric data.
          * Discrete numeric columns: numeric columns with fewer unique values than the `tresh` parameter (default=10).
          * Continuous numeric columns: numeric columns with unique values equal to or exceeding `tresh` (default=10).
      - For each column:
          * For categorical/discrete columns, a count plot is generated along with a detailed table
            of statistics and value counts.
          * For continuous columns, a statistics table is generated along with a boxplot (annotated with outliers)
            and an additional QQ plot to assess normality.
      - A summary page displaying overall statistics including the counts of each column type.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the dataset to be analyzed.
    tresh : int, optional
        The threshold for determining discrete versus continuous numeric columns. Numeric columns with
        a unique value count less than `tresh` are considered discrete; otherwise, they are considered continuous.
        Default is 10.
    output_file : str, optional
        The filename for the generated PDF report. Default is "report.pdf".
    df_name : str, optional
        A descriptive name for the DataFrame or dataset to be displayed in the report. Default is "Dataset".

    Returns
    -------
    None

    
    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "A": [1, 2, 3, 4, 5],
    ...     "B": ["a", "b", "a", "c", "b"],
    ...     "C": [10, 20, 30, 40, 50]
    ... })
    >>> generate_pdf_report_v5(df, tresh=3, output_file="example_report.pdf", df_name="Example Dataset")
    Report generated: example_report.pdf
    """


    # Define A4 page dimensions and margins
    PAGE_WIDTH, PAGE_HEIGHT = A4      # (595.27, 841.89) approximately
    margin_left = 50
    margin_right = 50
    margin_top = 50
    margin_bottom = 50
    content_width = PAGE_WIDTH - margin_left - margin_right

    # Create a new PDF canvas with A4 pagesize.
    c = canvas.Canvas(output_file, pagesize=A4)
    
    # --- Attempt to convert each column to numeric where possible ---
    for col in df.columns:
        # Convert values that can be interpreted as numbers; others become NaN.
        converted = pd.to_numeric(df[col], errors='coerce')
        if not converted.isna().all():
            df[col] = converted
    # ------------------------------------------------------------------

    # Categorize columns based on type and unique values.
    categorical_cols = []
    discrete_cols = []
    continuous_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < tresh:
                discrete_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            categorical_cols.append(col)
    
    # Identify columns with acceptable missingness (< 50% missing)
    low_missing_cols = [
        col for col in df.columns
        if (df[col].isna().sum() / len(df) if len(df) > 0 else 0) < 0.5
    ]
    
    # Version information for key libraries.
    version_info = {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'reportlab': '4.0.4',
    }
    
    # ----------------- COVER PAGE (A4) -----------------
    # Set title and subtitle positions using A4 dimensions.
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(PAGE_WIDTH/2, PAGE_HEIGHT - margin_top, "Data Analysis Report")
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH/2, PAGE_HEIGHT - margin_top - 30, f"Generated using AutoStats {auto_version}")
    
    # --- ASCII Art for "AutoStats" ---
    # Properly format the multi-line string into a list of lines
    ascii_logo_str = """
    *   _          _        ____  _        _       
      / \\   _ __ | |_ ___ / ___|| |_ __ _| |_ ___ 
     / _ \\ | | | | __/ _ \\___ \\| __/ _` | __/ __|
    / ___ \\| |_| | || (_) |__) | || (_| | |_\\__ \\
   /_/   \\_\\__,_|\\__\\___/____/ \\__\\__,_|\\__|___/
    """
    ascii_logo_lines = textwrap.dedent(ascii_logo_str).strip().split('\n')

    # Set font and position for the ASCII art
    c.setFont("Courier-Bold", 10)
    line_height = 12
    # Position the top of the ASCII art block
    ascii_y_start = PAGE_HEIGHT - margin_top - 150
    y_pos = ascii_y_start

    # Draw each line of the ASCII art, centered
    for line in ascii_logo_lines:
        c.drawCentredString(PAGE_WIDTH / 2, y_pos, line)
        y_pos -= line_height

    # Position for the metadata, below the ASCII art.
    metadata_start_y = y_pos - 40

    # Write metadata below the ASCII art.
    c.setFont("Helvetica", 12)
    metadata = [
        f"DataFrame Name: {df_name}",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Analysis Time: {datetime.now().strftime('%H:%M:%S')}",
        "",
        "Library Versions:",
        *[f"{lib}: {ver}" for lib, ver in version_info.items()]
    ]
    meta_y = metadata_start_y
    for line in metadata:
        c.drawString(margin_left, meta_y, line)
        meta_y -= 20

    c.showPage()  # End cover page.

    # ----------------- MAIN CONTENT (Each Column) -----------------
    # Define header positions for main content pages.
    header_y = PAGE_HEIGHT - margin_top  # Top of the page minus the top margin

    for column in df.columns:
        if column in categorical_cols or column in discrete_cols:
            # --- Process categorical/discrete columns ---
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin_left, header_y, "Column Analysis")
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin_left, header_y - 20, f"Column: {column}")
            y_position = header_y - 40  # Start content a bit below header

            # Compute basic statistics.
            nulls = df[column].isna().sum()
            missingness = nulls / len(df) if len(df) > 0 else 0
            total_count = len(df)
            value_counts = df[column].value_counts(dropna=True)
            total_non_missing = value_counts.sum()
            unique_count = len(value_counts)

            # Prepare statistics table.
            stats_data = [
                ["Statistic", "Value"],
                ["Missing Values", nulls],
                ["Missingness", f"{missingness:.2%}"],
                ["Total Count", total_count],
                ["Non-Missing Count", total_non_missing],
                ["Unique Values", unique_count],
                ["Value Counts", ""]
            ]
            max_values = 10
            for value, count in value_counts.head(max_values).items():
                percentage = (count / total_non_missing) * 100 if total_non_missing != 0 else 0
                stats_data.append([f"  {value}", f"{count} ({percentage:.1f}%)"])
            if len(value_counts) > max_values:
                stats_data.append([f"  ... (showing top {max_values})", f"{len(value_counts) - max_values} more"])

            # Create and style the table.
            col_widths = [120, 80]
            table = Table(stats_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            table.wrapOn(c, content_width, 200)
            table.drawOn(c, margin_left, y_position - table._height - 20)
            y_position -= table._height + 40

            # Generate a count plot for the column.
            plt.figure(figsize=(8, 4))
            value_counts_plot = df[column].value_counts()
            value_counts_plot.index = value_counts_plot.index.astype(str)
            bars = plt.bar(value_counts_plot.index, value_counts_plot.values)
            plt.title(f'Value Distribution: {column}')
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}',
                         ha='center', va='bottom', fontsize=8)
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            img = ImageReader(buffer)
            plt.close()

            max_img_height = y_position - margin_bottom
            img_width, img_height = img.getSize()
            scaling_factor = min(content_width / img_width, max_img_height / img_height)
            scaled_width = img_width * scaling_factor
            scaled_height = img_height * scaling_factor
            img_y = y_position - scaled_height
            c.drawImage(img, margin_left, img_y, width=scaled_width, height=scaled_height)
            c.setFont('Helvetica', 6)
            c.drawString(10, 10, f'Report generated by AutoStats library version {auto_version}')
            c.showPage()
        
        else:
            # --- Process continuous columns ---
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin_left, header_y, "Column Analysis")
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin_left, header_y - 20, f"Column: {column} (continuous variable)")
            y_position = header_y - 40

            # Compute basic statistics and perform normality test.
            stats = df[column].describe()
            nulls = df[column].isna().sum()
            uniques = df[column].nunique()
            missingness = nulls / len(df) if len(df) > 0 else 0
            _, nl_p = shapiro(df[column].dropna())
            nl = nl_p > 0.05

            # Prepare the statistics table.
            stats_data = [
                ["Statistic", "Value"],
                ["Missing Values", nulls],
                ["Missingness", f"{missingness:.2%}"],
                ["Unique Values", uniques],
                ['Shapiro Wilk test', f'{nl_p:.3f}'],
                ['Normal distribution', nl]
            ]
            stats_data += [[stat.capitalize(), f"{value:.2f}"] for stat, value in stats.items()]
            col_widths = [120, 80]
            table = Table(stats_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            table.wrapOn(c, content_width, 200)
            table.drawOn(c, margin_left, y_position - table._height - 20)
            y_position -= table._height + 40

            # Generate a boxplot for continuous data.
            plt.figure(figsize=(6, 3), dpi=200)
            clean_data = df[column].dropna()
            outliers_text = ""
            if not clean_data.empty:
                q1 = clean_data.quantile(0.25)
                q3 = clean_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                plt.boxplot(clean_data, vert=False)
                plt.title(f'Box Plot: {column}\n({len(outliers)} Outliers Detected)')
                plt.yticks([])
                if not outliers.empty:
                    outlier_indices = outliers.index.tolist()
                    displayed_outliers = [f"Index: {idx}, Value: {outliers[idx]:.2f}" for idx in outlier_indices[:15]]
                    outliers_text = "Key Outliers:\n" + "\n".join(displayed_outliers)
                    if len(outliers) > 15:
                        outliers_text += f"\n(Showing first 15 of {len(outliers)} outliers)"
                    for i, (idx, val) in enumerate(outliers.items()):
                        y_offset = 1.1 + (0.1 * (i % 2))
                        plt.annotate(
                            f'{idx}',
                            xy=(val, 1),
                            xytext=(val, y_offset),
                            textcoords='data',
                            arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
                            fontsize=6,
                            ha='center',
                            va='bottom',
                            rotation=45
                        )
            else:
                plt.text(0.5, 0.5, 'No valid data points', ha='center', va='center')
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            box_img = ImageReader(buffer)
            plt.close()

            max_img_height = y_position - margin_bottom
            img_width, img_height = box_img.getSize()
            scaling_factor = min(content_width / img_width, max_img_height / img_height)
            scaled_width = img_width * scaling_factor
            scaled_height = img_height * scaling_factor
            img_y = y_position - scaled_height
            c.drawImage(box_img, margin_left, img_y, width=scaled_width, height=scaled_height)

            # Generate a QQ plot and position it to the right and above the boxplot.
            if not clean_data.empty:
                plt.figure(figsize=(5, 5), dpi=200)
                probplot(clean_data, dist="norm", plot=plt)
                plt.title("QQ Plot")
                plt.tight_layout()
                buffer_qq = io.BytesIO()
                plt.savefig(buffer_qq, format='png', bbox_inches='tight', dpi=150)
                buffer_qq.seek(0)
                qq_img = ImageReader(buffer_qq)
                plt.close()
                
                qq_img_width, qq_img_height = qq_img.getSize()
                qq_scaling_factor = min(200 / qq_img_width, 200 / qq_img_height)
                scaled_qq_width = qq_img_width * qq_scaling_factor
                scaled_qq_height = qq_img_height * qq_scaling_factor
                # Position the QQ plot so that it is right-aligned with the boxplot and 35 points above it.
                qq_x = margin_left + scaled_width - scaled_qq_width + 10
                qq_y = img_y + scaled_height + 35
                c.drawImage(qq_img, qq_x, qq_y, width=scaled_qq_width, height=scaled_qq_height)

            # If any outlier annotation text exists, add it below the image.
            if outliers_text:
                text_y = img_y - 20
                text_obj = c.beginText(margin_left, text_y)
                text_obj.setFont("Helvetica", 8)
                text_obj.setLeading(9)
                for line in outliers_text.split('\n'):
                    text_obj.textLine(line)
                c.drawText(text_obj)
            c.setFont('Helvetica', 6)
            c.drawString(10, 10, f'Report generated by AutoStats library version {auto_version}')
            c.showPage()

    # ----------------- SUMMARY PAGE -----------------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_left, PAGE_HEIGHT - margin_top, "Summary Statistics")
    
    summary_data = [
        ["Metric", "Count"],
        ["Total Columns", len(df.columns)],
        ["Categorical Columns", len(categorical_cols)],
        ["Discrete columns", len(discrete_cols)],
        ["Continuous columns", len(continuous_cols)],
        ["Acceptable-Missingness Columns", len(low_missing_cols)]
    ]
    
    summary_table = Table(summary_data, colWidths=[200, 100])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    summary_table.wrapOn(c, content_width, 200)
    summary_table.drawOn(c, margin_left, PAGE_HEIGHT - margin_top - 150)
    c.setFont('Helvetica', 6)
    c.drawString(10, 10, f'Report generated by AutoStats library version {auto_version}')
    c.showPage()
    c.save()
    print(f"Report generated: {output_file}")

def manual_report(df: pd.DataFrame, categorical_cols:list, continuous_cols:list, discrete_cols:list=[],
                  output_file: str = "report.pdf", df_name: str = "Dataset") -> None:
    """
    Generate a PDF report for data exploration and analysis resized for A4 paper.

    The report includes:
      - A cover page with an ASCII art logo, metadata (DataFrame name, analysis date/time),
        and library version information.
      
      - For each column:
          * For categorical/discrete columns, a count plot is generated along with a detailed table
            of statistics and value counts.
          * For continuous columns, a statistics table is generated along with a boxplot (annotated with outliers)
            and an additional QQ plot to assess normality.
      - A summary page displaying overall statistics including the counts of each column type.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the dataset to be analyzed.
    categorical_cols : list
        A list of all the categorical columns 
    continuous_cols : list 
        A list of all the columns with continuous values 
    discrete_cols : list, optional 
        A list of the columns with numerical discrete values 
    output_file : str, optional
        The filename for the generated PDF report. Default is "report.pdf".
    df_name : str, optional
        A descriptive name for the DataFrame or dataset to be displayed in the report. Default is "Dataset".

    Returns
    -------
    None
    """
    # Define A4 page dimensions and margins
    PAGE_WIDTH, PAGE_HEIGHT = A4      # (595.27, 841.89) approximately
    margin_left = 50
    margin_right = 50
    margin_top = 50
    margin_bottom = 50
    content_width = PAGE_WIDTH - margin_left - margin_right

    # Create a new PDF canvas with A4 pagesize.
    c = canvas.Canvas(output_file, pagesize=A4)
    
    # --- Attempt to convert each column to numeric where possible ---
    for col in df.columns:
        # Convert values that can be interpreted as numbers; others become NaN.
        converted = pd.to_numeric(df[col], errors='coerce')
        if not converted.isna().all():
            df[col] = converted
    # ------------------------------------------------------------------
    # Identify columns with acceptable missingness (< 50% missing)
    low_missing_cols = [
        col for col in df.columns
        if (df[col].isna().sum() / len(df) if len(df) > 0 else 0) < 0.5
    ]
    
    # Version information for key libraries.
    version_info = {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'reportlab': '4.0.4',
    }
    
    # ----------------- COVER PAGE (A4) -----------------
    # Set title and subtitle positions using A4 dimensions.
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(PAGE_WIDTH/2, PAGE_HEIGHT - margin_top, "Data Analysis Report")
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_WIDTH/2, PAGE_HEIGHT - margin_top - 30, f"Generated using AutoStats {auto_version}")
    
    # --- ASCII Art for "AutoStats" ---
    # Properly format the multi-line string into a list of lines
    ascii_logo_str = """
    *   _          _        ____  _        _       
      / \\   _ __ | |_ ___ / ___|| |_ __ _| |_ ___ 
     / _ \\ | | | | __/ _ \\___ \\| __/ _` | __/ __|
    / ___ \\| |_| | || (_) |__) | || (_| | |_\\__ \\
   /_/   \\_\\__,_|\\__\\___/____/ \\__\\__,_|\\__|___/
    """
    ascii_logo_lines = textwrap.dedent(ascii_logo_str).strip().split('\n')

    # Set font and position for the ASCII art
    c.setFont("Courier-Bold", 10)
    line_height = 12
    # Position the top of the ASCII art block
    ascii_y_start = PAGE_HEIGHT - margin_top - 150
    y_pos = ascii_y_start

    # Draw each line of the ASCII art, centered
    for line in ascii_logo_lines:
        c.drawCentredString(PAGE_WIDTH / 2, y_pos, line)
        y_pos -= line_height

    # Position for the metadata, below the ASCII art.
    metadata_start_y = y_pos - 40

    # Write metadata below the ASCII art.
    c.setFont("Helvetica", 12)
    metadata = [
        f"DataFrame Name: {df_name}",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Analysis Time: {datetime.now().strftime('%H:%M:%S')}",
        "",
        "Library Versions:",
        *[f"{lib}: {ver}" for lib, ver in version_info.items()]
    ]
    meta_y = metadata_start_y
    for line in metadata:
        c.drawString(margin_left, meta_y, line)
        meta_y -= 20

    c.showPage()  # End cover page.

    # ----------------- MAIN CONTENT (Each Column) -----------------
    # Define header positions for main content pages.
    header_y = PAGE_HEIGHT - margin_top  # Top of the page minus the top margin

    for column in df.columns:
        if column in categorical_cols or column in discrete_cols:
            # --- Process categorical/discrete columns ---
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin_left, header_y, "Column Analysis")
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin_left, header_y - 20, f"Column: {column}")
            y_position = header_y - 40  # Start content a bit below header

            # Compute basic statistics.
            nulls = df[column].isna().sum()
            missingness = nulls / len(df) if len(df) > 0 else 0
            total_count = len(df)
            value_counts = df[column].value_counts(dropna=True)
            total_non_missing = value_counts.sum()
            unique_count = len(value_counts)

            # Prepare statistics table.
            stats_data = [
                ["Statistic", "Value"],
                ["Missing Values", nulls],
                ["Missingness", f"{missingness:.2%}"],
                ["Total Count", total_count],
                ["Non-Missing Count", total_non_missing],
                ["Unique Values", unique_count],
                ["Value Counts", ""]
            ]
            max_values = 10
            for value, count in value_counts.head(max_values).items():
                percentage = (count / total_non_missing) * 100 if total_non_missing != 0 else 0
                stats_data.append([f"  {value}", f"{count} ({percentage:.1f}%)"])
            if len(value_counts) > max_values:
                stats_data.append([f"  ... (showing top {max_values})", f"{len(value_counts) - max_values} more"])

            # Create and style the table.
            col_widths = [120, 80]
            table = Table(stats_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            table.wrapOn(c, content_width, 200)
            table.drawOn(c, margin_left, y_position - table._height - 20)
            y_position -= table._height + 40

            # Generate a count plot for the column.
            plt.figure(figsize=(8, 4))
            value_counts_plot = df[column].value_counts()
            value_counts_plot.index = value_counts_plot.index.astype(str)
            bars = plt.bar(value_counts_plot.index, value_counts_plot.values)
            plt.title(f'Value Distribution: {column}')
            plt.xticks(rotation=45)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}',
                         ha='center', va='bottom', fontsize=8)
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            img = ImageReader(buffer)
            plt.close()

            max_img_height = y_position - margin_bottom
            img_width, img_height = img.getSize()
            scaling_factor = min(content_width / img_width, max_img_height / img_height)
            scaled_width = img_width * scaling_factor
            scaled_height = img_height * scaling_factor
            img_y = y_position - scaled_height
            c.drawImage(img, margin_left, img_y, width=scaled_width, height=scaled_height)
            c.setFont('Helvetica', 6)
            c.drawString(10, 10, f'Report generated by AutoStats library version {auto_version}')
            c.showPage()
        
        else:
            # --- Process continuous columns ---
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin_left, header_y, "Column Analysis")
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin_left, header_y - 20, f"Column: {column} (continuous variable)")
            y_position = header_y - 40

            # Compute basic statistics and perform normality test.
            stats = df[column].describe()
            nulls = df[column].isna().sum()
            uniques = df[column].nunique()
            missingness = nulls / len(df) if len(df) > 0 else 0
            _, nl_p = shapiro(df[column].dropna())
            nl = nl_p > 0.05

            # Prepare the statistics table.
            stats_data = [
                ["Statistic", "Value"],
                ["Missing Values", nulls],
                ["Missingness", f"{missingness:.2%}"],
                ["Unique Values", uniques],
                ['Shapiro Wilk test', f'{nl_p:.3f}'],
                ['Normal distribution', nl]
            ]
            stats_data += [[stat.capitalize(), f"{value:.2f}"] for stat, value in stats.items()]
            col_widths = [120, 80]
            table = Table(stats_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            table.wrapOn(c, content_width, 200)
            table.drawOn(c, margin_left, y_position - table._height - 20)
            y_position -= table._height + 40

            # Generate a boxplot for continuous data.
            plt.figure(figsize=(6, 3), dpi=200)
            clean_data = df[column].dropna()
            outliers_text = ""
            if not clean_data.empty:
                q1 = clean_data.quantile(0.25)
                q3 = clean_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                plt.boxplot(clean_data, vert=False)
                plt.title(f'Box Plot: {column}\n({len(outliers)} Outliers Detected)')
                plt.yticks([])
                if not outliers.empty:
                    outlier_indices = outliers.index.tolist()
                    displayed_outliers = [f"Index: {idx}, Value: {outliers[idx]:.2f}" for idx in outlier_indices[:15]]
                    outliers_text = "Key Outliers:\n" + "\n".join(displayed_outliers)
                    if len(outliers) > 15:
                        outliers_text += f"\n(Showing first 15 of {len(outliers)} outliers)"
                    for i, (idx, val) in enumerate(outliers.items()):
                        y_offset = 1.1 + (0.1 * (i % 2))
                        plt.annotate(
                            f'{idx}',
                            xy=(val, 1),
                            xytext=(val, y_offset),
                            textcoords='data',
                            arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
                            fontsize=6,
                            ha='center',
                            va='bottom',
                            rotation=45
                        )
            else:
                plt.text(0.5, 0.5, 'No valid data points', ha='center', va='center')
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            box_img = ImageReader(buffer)
            plt.close()

            max_img_height = y_position - margin_bottom
            img_width, img_height = box_img.getSize()
            scaling_factor = min(content_width / img_width, max_img_height / img_height)
            scaled_width = img_width * scaling_factor
            scaled_height = img_height * scaling_factor
            img_y = y_position - scaled_height
            c.drawImage(box_img, margin_left, img_y, width=scaled_width, height=scaled_height)

            # Generate a QQ plot and position it to the right and above the boxplot.
            if not clean_data.empty:
                plt.figure(figsize=(5, 5), dpi=200)
                probplot(clean_data, dist="norm", plot=plt)
                plt.title("QQ Plot")
                plt.tight_layout()
                buffer_qq = io.BytesIO()
                plt.savefig(buffer_qq, format='png', bbox_inches='tight', dpi=150)
                buffer_qq.seek(0)
                qq_img = ImageReader(buffer_qq)
                plt.close()
                
                qq_img_width, qq_img_height = qq_img.getSize()
                qq_scaling_factor = min(200 / qq_img_width, 200 / qq_img_height)
                scaled_qq_width = qq_img_width * qq_scaling_factor
                scaled_qq_height = qq_img_height * qq_scaling_factor
                # Position the QQ plot so that it is right-aligned with the boxplot and 35 points above it.
                qq_x = margin_left + scaled_width - scaled_qq_width + 10
                qq_y = img_y + scaled_height + 35
                c.drawImage(qq_img, qq_x, qq_y, width=scaled_qq_width, height=scaled_qq_height)

            # If any outlier annotation text exists, add it below the image.
            if outliers_text:
                text_y = img_y - 20
                text_obj = c.beginText(margin_left, text_y)
                text_obj.setFont("Helvetica", 8)
                text_obj.setLeading(9)
                for line in outliers_text.split('\n'):
                    text_obj.textLine(line)
                c.drawText(text_obj)
            c.setFont('Helvetica', 6)
            c.drawString(10, 10, f'Report generated by AutoStats library version {auto_version}')
            c.showPage()

    # ----------------- SUMMARY PAGE -----------------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin_left, PAGE_HEIGHT - margin_top, "Summary Statistics")
    
    summary_data = [
        ["Metric", "Count"],
        ["Total Columns", len(df.columns)],
        ["Categorical Columns", len(categorical_cols)],
        ["Discrete columns", len(discrete_cols)],
        ["Continuous columns", len(continuous_cols)],
        ["Acceptable-Missingness Columns", len(low_missing_cols)]
    ]
    
    summary_table = Table(summary_data, colWidths=[200, 100])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    summary_table.wrapOn(c, content_width, 200)
    summary_table.drawOn(c, margin_left, PAGE_HEIGHT - margin_top - 150)
    c.setFont('Helvetica', 6)
    c.drawString(10, 10, f'Report generated by AutoStats library version {auto_version}')
    c.showPage()
    c.save()
    print(f"Report generated: {output_file}")