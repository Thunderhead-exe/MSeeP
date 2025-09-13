"""
Model Context Protocol (MCP) Server for Mathematical Equation Plotting - ImgBB Upload Version

This server provides tools for extracting symbolic equations from text
and plotting them to enhance mathematical problem understanding.
This version uploads the generated plot to ImgBB and returns the hosted image URL.
"""

import os
import re
import base64
import tempfile
import requests
from typing import List, Dict, Any, Optional, Union

import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Math Equation Plotter Upload")

# Global storage for extracted equations (storing string expressions and SymPy objects)
extracted_equations: Dict[str, Dict[str, Any]] = {}


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
        # Implicit equations: x^2 + y^2 = 1
        r'([^=\n]+)\s*=\s*([^=\n]+)',
        # Function definitions: f(x) = x^2
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s*=\s*([^=\n]+)',
    ]
    
    for i, pattern in enumerate(patterns):
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if i == 0:  # Explicit equations
                    var_name, dep_var, expr_str = match.groups()
                    if dep_var:  # Function form: f(x) = ...
                        expr_str = match.group(3)
                        var_name = match.group(1)
                        dep_var = match.group(2)
                    else:  # Simple form: y = ...
                        expr_str = match.group(3)
                        var_name = match.group(1)
                        dep_var = None
                elif i == 1:  # Implicit equations
                    left_expr, right_expr = match.groups()
                    expr_str = f"({left_expr}) - ({right_expr})"
                    var_name = "implicit"
                    dep_var = None
                else:  # Function definitions
                    var_name, dep_var, expr_str = match.groups()
                
                # Clean and normalize the expression
                expr_str = clean_expression(expr_str)
                
                # Try to parse with SymPy
                try:
                    expr = sp.sympify(expr_str)
                    equation_id = f"eq{len(equations) + 1}"
                    
                    equations.append({
                        'id': equation_id,
                        'name': var_name,
                        'expression': str(expr),
                        'dependent_var': dep_var,
                        'type': 'function' if dep_var else 'explicit',
                        'original_text': match.group(0)
                    })
                    
                    # Store in global dictionary (both string and SymPy object)
                    extracted_equations[equation_id] = {
                        'expression': str(expr),
                        'sympy_expr': expr,
                        'latex': sp.latex(expr),
                        'variables': [str(sym) for sym in expr.free_symbols]
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not parse expression '{expr_str}': {e}")
                    continue
                    
            except Exception as e:
                print(f"Warning: Error processing match '{match.group(0)}': {e}")
                continue
    
    return equations


def clean_expression(expr_str: str) -> str:
    """
    Clean and normalize mathematical expressions for SymPy parsing.
    
    Args:
        expr_str (str): Raw expression string
        
    Returns:
        str: Cleaned expression string
    """
    # Remove extra whitespace
    expr_str = re.sub(r'\s+', ' ', expr_str.strip())
    
    # Convert common mathematical notation
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
    
    # Handle implicit multiplication (e.g., "2x" -> "2*x")
    # But avoid adding * after function names
    expr_str = re.sub(r'(\d+)([a-zA-Z_])', r'\1*\2', expr_str)
    expr_str = re.sub(r'([a-zA-Z_]+)(\d+)', r'\1*\2', expr_str)
    expr_str = re.sub(r'\)([a-zA-Z_])', r')*\1', expr_str)
    
    # Fix common function parsing issues
    expr_str = re.sub(r'si\*n', 'sin', expr_str)
    expr_str = re.sub(r'co\*s', 'cos', expr_str)
    expr_str = re.sub(r'ta\*n', 'tan', expr_str)
    
    return expr_str


def plot_equations_to_imgbb(equation_ids: List[str], 
                           x_range: tuple = (-10, 10), 
                           y_range: Optional[tuple] = None,
                           resolution: int = 1000,
                           title: str = "Mathematical Functions") -> Dict[str, Any]:
    """
    Plot one or more extracted equations, save to temporary file, upload to ImgBB, and return the complete response.
    
    Args:
        equation_ids (List[str]): List of equation IDs to plot
        x_range (tuple): X-axis range (min, max)
        y_range (Optional[tuple]): Y-axis range (min, max), auto if None
        resolution (int): Number of points for plotting
        title (str): Plot title
        
    Returns:
        Dict[str, Any]: Complete ImgBB API response with image URLs and metadata
        
    Examples:
        >>> plot_equations_to_imgbb(['eq1'], x_range=(-5, 5))
        {'data': {'url': 'https://i.ibb.co/abc123/plot.png', ...}, 'success': True, ...}
    """
    if not equation_ids:
        raise ValueError("No equation IDs provided")
    
    # Validate equation IDs
    missing_ids = [eq_id for eq_id in equation_ids if eq_id not in extracted_equations]
    if missing_ids:
        raise ValueError(f"Equations not found: {missing_ids}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate x values
    x_min, x_max = x_range
    x_vals = np.linspace(x_min, x_max, resolution)
    
    # Plot each equation
    colors = plt.cm.tab10(np.linspace(0, 1, len(equation_ids)))
    
    for i, eq_id in enumerate(equation_ids):
        expr = extracted_equations[eq_id]['sympy_expr']
        
        try:
            # Convert to numerical function
            if hasattr(expr, 'free_symbols'):
                symbols = list(expr.free_symbols)
                if len(symbols) == 1:
                    # Single variable function
                    var = symbols[0]
                    func = sp.lambdify(var, expr, 'numpy')
                    y_vals = func(x_vals)
                    
                    # Handle complex results
                    if np.iscomplexobj(y_vals):
                        y_vals = np.real(y_vals)
                    
                    # Plot
                    ax.plot(x_vals, y_vals, 
                           color=colors[i], 
                           linewidth=2, 
                           label=f'{eq_id}: {sp.latex(expr)}')
                    
                elif len(symbols) == 2:
                    # Implicit equation (contour plot)
                    x_sym, y_sym = symbols[:2]
                    y_min, y_max = y_range if y_range else x_range
                    y_vals_2d = np.linspace(y_min, y_max, resolution)
                    X, Y = np.meshgrid(x_vals, y_vals_2d)
                    Z = sp.lambdify((x_sym, y_sym), expr, 'numpy')(X, Y)
                    
                    ax.contour(X, Y, Z, levels=[0], colors=[colors[i]], linewidths=2)
                    ax.plot([], [], color=colors[i], linewidth=2, 
                           label=f'{eq_id}: {sp.latex(expr)} = 0')
                    
        except Exception as e:
            print(f"Warning: Could not plot equation {eq_id}: {e}")
            continue
    
    # Customize plot
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set axis limits
    ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)
    else:
        ax.set_ylim(ax.get_ylim())  # Auto-scale
    
    # Add axis lines
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    try:
        # Upload to ImgBB
        imgbb_response = upload_to_imgbb(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return imgbb_response
        
    except Exception as e:
        # Clean up temporary file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e


def upload_to_imgbb(image_path: str) -> Dict[str, Any]:
    """
    Upload an image file to ImgBB and return the complete response data.
    
    Args:
        image_path (str): Path to the image file to upload
        
    Returns:
        Dict[str, Any]: Complete ImgBB API response with image URLs and metadata
        
    Raises:
        ValueError: If API key is not set or upload fails
        requests.RequestException: If HTTP request fails
    """
    # Get API key from environment
    api_key = os.environ.get("IMGBB_CLIENT_API_KEY")
    if not api_key:
        raise ValueError("IMGBB_CLIENT_API_KEY environment variable not set")
    
    # Read and encode image file
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Prepare upload request
    url = "https://api.imgbb.com/1/upload"
    params = {
        "expiration": 600,  # 10 minutes expiration
        "key": api_key
    }
    data = {
        "image": image_data
    }
    
    # Upload to ImgBB
    response = requests.post(url, params=params, data=data)
    response.raise_for_status()
    
    # Parse response
    result = response.json()
    
    if result.get("success"):
        return result
    else:
        raise ValueError(f"ImgBB upload failed: {result.get('error', {}).get('message', 'Unknown error')}")


# MCP Tool Definitions
@mcp.tool(
    title="Extract Mathematical Equations",
    description="Extract symbolic equations from natural language text containing mathematical expressions. Supports polynomials, trigonometric functions, exponential functions, and implicit equations."
)
def extract_equations(text: str) -> Dict[str, Any]:
    """
    Extract symbolic equations from input text.
    
    Args:
        text (str): Input text containing mathematical equations
        
    Returns:
        Dict[str, Any]: Extracted equations with metadata
        
    Examples:
        - "y = x^2 + 3x + 2"
        - "f(x) = sin(x) + cos(x)"
        - "x^2 + y^2 = 1"
    """
    try:
        equations = extract_equations_from_text(text)
        return {
            "success": True,
            "equations": equations,
            "count": len(equations),
            "message": f"Successfully extracted {len(equations)} equation(s)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to extract equations"
        }


@mcp.tool(
    title="Plot Mathematical Equations and Upload to ImgBB",
    description="Generate high-quality plots of extracted mathematical equations, upload them to ImgBB, and return the complete ImgBB API response with image URLs and metadata. Supports single and multiple equation plotting with customizable ranges."
)
def plot_extracted_equations(equation_ids: List[str],
                           x_range: List[float] = [-10, 10],
                           y_range: Optional[List[float]] = None,
                           resolution: int = 1000,
                           title: str = "Mathematical Functions") -> Dict[str, Any]:
    """
    Plot extracted equations, upload to ImgBB, and return the complete ImgBB API response.
    
    Args:
        equation_ids (List[str]): List of equation IDs to plot
        x_range (List[float]): X-axis range [min, max]
        y_range (Optional[List[float]]): Y-axis range [min, max]
        resolution (int): Number of points for plotting
        title (str): Plot title
        
    Returns:
        Dict[str, Any]: Complete ImgBB API response with image URLs and metadata
    """
    try:
        x_range_tuple = tuple(x_range)
        y_range_tuple = tuple(y_range) if y_range else None
        
        imgbb_response = plot_equations_to_imgbb(
            equation_ids=equation_ids,
            x_range=x_range_tuple,
            y_range=y_range_tuple,
            resolution=resolution,
            title=title
        )
        
        # Add our custom metadata to the ImgBB response
        imgbb_response["plot_metadata"] = {
            "equation_ids": equation_ids,
            "x_range": x_range,
            "y_range": y_range,
            "title": title,
            "message": f"Successfully plotted {len(equation_ids)} equation(s) and uploaded to ImgBB"
        }
        
        # Add the canvas iframe format
        image_url = imgbb_response["data"]["url"]
        imgbb_response["canvas_iframe"] = f"show this image in canvas using iframe, url={image_url}"
        
        return imgbb_response
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to plot equations and upload to ImgBB"
        }


@mcp.tool(
    title="List Extracted Equations",
    description="List all currently extracted mathematical equations stored in memory, including their expressions, LaTeX representations, and variable information."
)
def list_extracted_equations() -> Dict[str, Any]:
    """
    List all currently extracted equations.
    
    Returns:
        Dict[str, Any]: List of extracted equations
    """
    equations_info = []
    for eq_id, eq_data in extracted_equations.items():
        equations_info.append({
            "id": eq_id,
            "expression": eq_data['expression'],
            "latex": eq_data['latex'],
            "variables": eq_data['variables']
        })
    
    return {
        "success": True,
        "equations": equations_info,
        "count": len(equations_info),
        "message": f"Found {len(equations_info)} extracted equation(s)"
    }


@mcp.tool(
    title="Clear Extracted Equations",
    description="Clear all extracted mathematical equations from memory. Useful for starting fresh or managing memory usage when working with multiple equation sets."
)
def clear_equations() -> Dict[str, Any]:
    """
    Clear all extracted equations from memory.
    
    Returns:
        Dict[str, Any]: Confirmation message
    """
    global extracted_equations
    count = len(extracted_equations)
    extracted_equations.clear()
    
    return {
        "success": True,
        "message": f"Cleared {count} equation(s) from memory"
    }
