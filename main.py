"""
Model Context Protocol (MCP) Server for Mathematical Equation Plotting - Base64 Version

This server provides tools for extracting symbolic equations from text
and plotting them to enhance mathematical problem understanding.
This version returns base64 data directly without saving files.
"""

import os
import re
import base64
import io
from typing import List, Dict, Any, Optional, Union

import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from mcp.server.fastmcp import FastMCP
from pydantic import Field

import mcp.types as types

# Initialize FastMCP server
mcp = FastMCP("Math Equation Plotter", port=3003, stateless_http=True, debug=True)


# Global storage for extracted equations
extracted_equations: Dict[str, sp.Expr] = {}


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
                        'sympy_expr': expr,
                        'dependent_var': dep_var,
                        'type': 'function' if dep_var else 'explicit',
                        'original_text': match.group(0)
                    })
                    
                    # Store in global dictionary
                    extracted_equations[equation_id] = expr
                    
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
    
    # Only add * before parentheses if it's not a function call
    # This is tricky - we'll be more conservative and only handle obvious cases
    # expr_str = re.sub(r'([a-zA-Z_])(\()', r'\1*\2', expr_str)  # Removed this line
    
    return expr_str


def plot_equations_to_base64(equation_ids: List[str], 
                            x_range: tuple = (-10, 10), 
                            y_range: Optional[tuple] = None,
                            resolution: int = 1000,
                            title: str = "Mathematical Functions") -> str:
    """
    Plot one or more extracted equations and return the image as base64.
    This version does not save files, only returns base64 data.
    
    Args:
        equation_ids (List[str]): List of equation IDs to plot
        x_range (tuple): X-axis range (min, max)
        y_range (Optional[tuple]): Y-axis range (min, max), auto if None
        resolution (int): Number of points for plotting
        title (str): Plot title
        
    Returns:
        str: Base64 encoded image data
        
    Examples:
        >>> plot_equations_to_base64(['eq1'], x_range=(-5, 5))
        'iVBORw0KGgoAAAANSUhEUgAA...'  # Base64 image data
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
        expr = extracted_equations[eq_id]
        
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
    
    # Save to BytesIO buffer instead of file
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Get the image data and convert to base64
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    
    return image_data


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
    title="Plot Mathematical Equations",
    description="Generate high-quality plots of extracted mathematical equations and return them as base64-encoded PNG images. Supports single and multiple equation plotting with customizable ranges."
)
def plot_extracted_equations(equation_ids: List[str],
                           x_range: List[float] = [-10, 10],
                           y_range: Optional[List[float]] = None,
                           resolution: int = 1000,
                           title: str = "Mathematical Functions") -> Dict[str, Any]:
    """
    Plot extracted equations and return the image as base64 data.
    This version does not save files to disk.
    
    Args:
        equation_ids (List[str]): List of equation IDs to plot
        x_range (List[float]): X-axis range [min, max]
        y_range (Optional[List[float]]): Y-axis range [min, max]
        resolution (int): Number of points for plotting
        title (str): Plot title
        
    Returns:
        Dict[str, Any]: Plot result with base64 image data
    """
    try:
        x_range_tuple = tuple(x_range)
        y_range_tuple = tuple(y_range) if y_range else None
        
        image_data = plot_equations_to_base64(
            equation_ids=equation_ids,
            x_range=x_range_tuple,
            y_range=y_range_tuple,
            resolution=resolution,
            title=title
        )
        
        return {
            "success": True,
            "image_data": image_data,
            "equation_ids": equation_ids,
            "message": f"Successfully plotted {len(equation_ids)} equation(s)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to plot equations"
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
    for eq_id, expr in extracted_equations.items():
        equations_info.append({
            "id": eq_id,
            "expression": str(expr),
            "latex": sp.latex(expr),
            "variables": [str(sym) for sym in expr.free_symbols]
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


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
