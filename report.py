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

auto_version = 'alpha_1.0'

