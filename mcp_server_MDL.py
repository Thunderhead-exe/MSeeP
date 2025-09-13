"""
Model Context Protocol (MCP) Server for Mathematical Equation Plotting - OCR Version

This server provides tools for extracting symbolic equations from text
and plotting them using Mistral's OCR library to process the images.
This version processes plots with OCR and returns the OCR results.
"""

import os
import re
import base64
import io
import tempfile
from typing import List, Dict, Any, Optional, Union

import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Math Equation Plotter OCR")

# Global storage for extracted equations (storing string expressions and SymPy objects)
extracted_equations: Dict[str, Dict[str, Any]] = {}


def encode_image(image_path: str) -> Optional[str]:
    """
    Encode the image to base64.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Optional[str]: Base64 encoded image string or None if error
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def extract_equations_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract symbolic equations from input text using pattern matching and SymPy parsing.
    
    Args:
        text (str): Input text containing mathematical equations
        
    Returns:
        List[Dict[str, Any]]: List of extracted equations with metadata
        
    Examples:
        >>> extract_equations_from_text("y = x^2 + 3x + 2")
        [{'name': 'eq1', 'expression': 'x**2 + 3*x + 2', 'variable': 'y', 'type': 'explicit'}]
        
        >>> extract_equations_from_text("f(x) = sin(x) + cos(x)")
        [{'name': 'eq1', 'expression': 'sin(x) + cos(x)', 'variable': 'f', 'type': 'function'}]
    """
    equations = []
    
    # Common equation patterns
    patterns = [
        # Explicit equations: y = x^2, f(x) = sin(x), etc.
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(?([a-zA-Z_][a-zA-Z0-9_]*)?\)?\s*=\s*([^=\n]+)',
        # Implicit equations: x^2 + y^2 = 1, etc.
        r'([^=\n]+)\s*=\s*([^=\n]+)',
        # Function definitions: f(x) = expression
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s*=\s*([^=\n]+)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if len(match.groups()) == 3:
                    # Function or explicit equation
                    var_name, dep_var, expr_str = match.groups()
                    if dep_var:  # Function definition
                        expr_str = f"{expr_str}"
                        dep_var = dep_var.strip()
                    else:  # Explicit equation
                        dep_var = None
                else:
                    # Implicit equation
                    left_expr, right_expr = match.groups()
                    expr_str = f"{left_expr} - ({right_expr})"
                    var_name = "implicit"
                    dep_var = None
                
                # Clean and parse expression
                cleaned_expr = clean_expression(expr_str)
                expr = sp.sympify(cleaned_expr)
                
                # Generate equation ID
                equation_id = f"eq{len(extracted_equations) + 1}"
                
                # Store in global dictionary
                extracted_equations[equation_id] = {
                    'expression': str(expr),
                    'sympy_expr': expr,
                    'latex': sp.latex(expr),
                    'variables': [str(sym) for sym in expr.free_symbols]
                }
                
                # Add to equations list
                equations.append({
                    'id': equation_id,
                    'name': var_name,
                    'expression': str(expr),
                    'dependent_var': dep_var,
                    'type': 'function' if dep_var else 'explicit',
                    'original_text': match.group(0)
                })
                
            except Exception as e:
                print(f"Warning: Could not parse expression '{match.group(0)}': {e}")
                continue
    
    return equations


def clean_expression(expr_str: str) -> str:
    """
    Clean mathematical expression string for SymPy parsing.
    
    Args:
        expr_str (str): Raw expression string
        
    Returns:
        str: Cleaned expression string
    """
    # Remove extra whitespace
    expr_str = re.sub(r'\s+', ' ', expr_str.strip())
    
    # Handle common mathematical notation
    replacements = {
        r'\^': '**',  # Power operator
        r'²': '**2',  # Superscript 2
        r'³': '**3',  # Superscript 3
        r'¹': '**1',  # Superscript 1
        r'⁰': '**0',  # Superscript 0
        r'\bpi\b': 'pi',  # Pi constant
        r'\be\b': 'E',  # Euler's number
        r'\bsin\b': 'sin',  # Trigonometric functions
        r'\bcos\b': 'cos',
        r'\btan\b': 'tan',
        r'\blog\b': 'log',
        r'\bln\b': 'log',
        r'\bsqrt\b': 'sqrt',
        r'\babs\b': 'Abs',
    }
    
    for pattern, replacement in replacements.items():
        expr_str = re.sub(pattern, replacement, expr_str, flags=re.IGNORECASE)
    
    # Handle implicit multiplication (but avoid function names)
    # Only add * between numbers and variables, not after function names
    expr_str = re.sub(r'(\d)([a-zA-Z_])', r'\1*\2', expr_str)
    # Avoid adding * after function names like sin, cos, etc.
    expr_str = re.sub(r'([a-zA-Z_]\w*)([a-zA-Z_]\w*)', r'\1*\2', expr_str)
    # Fix common function parsing issues
    expr_str = re.sub(r'si\*n', 'sin', expr_str)
    expr_str = re.sub(r'co\*s', 'cos', expr_str)
    expr_str = re.sub(r'ta\*n', 'tan', expr_str)
    
    return expr_str


def plot_equations_to_ocr(
    equation_ids: List[str],
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    title: Optional[str] = None,
    resolution: int = 1000
) -> Dict[str, Any]:
    """
    Plot extracted equations, save to temporary PNG file, and process with Mistral OCR.
    
    Args:
        equation_ids (List[str]): List of equation IDs to plot
        x_range (Optional[List[float]]): X-axis range [min, max]
        y_range (Optional[List[float]]): Y-axis range [min, max] (for implicit equations)
        title (Optional[str]): Plot title
        resolution (int): Plot resolution
        
    Returns:
        Dict[str, Any]: Plot result with OCR processing results
    """
    if not equation_ids:
        return {
            'success': False,
            'error': 'No equation IDs provided',
            'message': 'Please provide equation IDs to plot'
        }
    
    # Validate equation IDs
    missing_ids = [eq_id for eq_id in equation_ids if eq_id not in extracted_equations]
    if missing_ids:
        return {
            'success': False,
            'error': f'Equation IDs not found: {missing_ids}',
            'message': 'Please extract equations first'
        }
    
    # Set default ranges
    if x_range is None:
        x_range = [-5, 5]
    if y_range is None:
        y_range = [-5, 5]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot each equation
    for i, eq_id in enumerate(equation_ids):
        expr_data = extracted_equations[eq_id]
        expr = expr_data['sympy_expr']
        symbols = list(expr.free_symbols)
        
        if len(symbols) == 1:
            # Single variable function
            x_sym = symbols[0]
            x_vals = np.linspace(x_range[0], x_range[1], resolution)
            y_vals = sp.lambdify(x_sym, expr, 'numpy')(x_vals)
            ax.plot(x_vals, y_vals, color=colors[i % len(colors)], linewidth=2,
                   label=f'{eq_id}: {sp.latex(expr)}')
            
        elif len(symbols) == 2:
            # Implicit equation (contour plot)
            x_sym, y_sym = symbols[:2]
            x_vals = np.linspace(x_range[0], x_range[1], resolution)
            y_min, y_max = y_range if y_range else x_range
            y_vals_2d = np.linspace(y_min, y_max, resolution)
            X, Y = np.meshgrid(x_vals, y_vals_2d)
            Z = sp.lambdify((x_sym, y_sym), expr, 'numpy')(X, Y)
            ax.contour(X, Y, Z, levels=[0], colors=[colors[i % len(colors)]], linewidths=2)
            ax.plot([], [], color=colors[i % len(colors)], linewidth=2,
                   label=f'{eq_id}: {sp.latex(expr)} = 0')
    
    # Customize plot
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title or 'Mathematical Equations Plot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)
    
    # Add axis lines
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create temporary PNG file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Save plot to temporary file
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    try:
        # Initialize Mistral client
        from mistralai import Mistral
        
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        client = Mistral(api_key=api_key)
        
        # Encode image to base64
        base64_image = encode_image(temp_path)
        if not base64_image:
            raise ValueError("Failed to encode image to base64")
        
        # Process with OCR
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/png;base64,{base64_image}"
            },
            include_image_base64=False
        )
        
        
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return {
            'success': True,
            'title': title,
            'equation_ids': equation_ids,
            'x_range': x_range,
            'y_range': y_range,
            'ocr_results': ocr_response,
            'extracted_image': ocr_response.pages[0].images[0],
            'image_format': 'png',
            'message': f"Successfully processed plot with Mistral OCR and extracted image"
        }
        
    except Exception as e:
        # Clean up temporary file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to process plot with Mistral OCR"
        }


# MCP Tools
@mcp.tool(
    title="Extract Mathematical Equations",
    description="Extract symbolic equations from natural language text containing mathematical expressions. Supports polynomials, trigonometric functions, exponential functions, and implicit equations."
)
def extract_equations(text: str) -> Dict[str, Any]:
    """
    Extract mathematical equations from text.
    
    Args:
        text (str): Input text containing mathematical equations
        
    Returns:
        Dict[str, Any]: Extracted equations with metadata
    """
    try:
        equations = extract_equations_from_text(text)
        return {
            'success': True,
            'equations': equations,
            'count': len(equations),
            'message': f'Successfully extracted {len(equations)} equation(s)'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to extract equations'
        }


@mcp.tool(
    title="Plot Extracted Equations with OCR",
    description="Plot one or more extracted mathematical equations and process the resulting image with Mistral OCR to extract text and mathematical content from the plot. Returns the extracted image from OCR bounding box results."
)
def plot_extracted_equations(
    equation_ids: List[str],
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Plot extracted equations and process with OCR.
    
    Args:
        equation_ids (List[str]): List of equation IDs to plot
        x_range (Optional[List[float]]): X-axis range [min, max]
        y_range (Optional[List[float]]): Y-axis range [min, max]
        title (Optional[str]): Plot title
        
    Returns:
        Dict[str, Any]: Plot result with OCR processing results
    """
    return plot_equations_to_ocr(equation_ids, x_range, y_range, title)


@mcp.tool(
    title="List Extracted Equations",
    description="List all currently extracted mathematical equations with their IDs, expressions, and metadata."
)
def list_extracted_equations() -> Dict[str, Any]:
    """
    List all extracted equations.
    
    Returns:
        Dict[str, Any]: List of extracted equations
    """
    equations_info = []
    for eq_id, eq_data in extracted_equations.items():
        equations_info.append({
            'id': eq_id,
            'expression': eq_data['expression'],
            'latex': eq_data['latex'],
            'variables': eq_data['variables']
        })
    
    return {
        'success': True,
        'equations': equations_info,
        'count': len(equations_info),
        'message': f'Found {len(equations_info)} extracted equation(s)'
    }


@mcp.tool(
    title="Clear Extracted Equations",
    description="Clear all extracted mathematical equations from memory. Useful for starting fresh or managing memory usage when working with multiple equation sets."
)
def clear_equations() -> Dict[str, Any]:
    """
    Clear all extracted equations.
    
    Returns:
        Dict[str, Any]: Clear operation result
    """
    global extracted_equations
    count = len(extracted_equations)
    extracted_equations.clear()
    
    return {
        'success': True,
        'count': count,
        'message': f'Cleared {count} equation(s) from memory'
    }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(port=8000, stateless_http=True, debug=True)
