"""
Model Context Protocol (MCP) Server for Mathematical Equation Plotting - Mistral Document Library Version

This server provides tools for extracting symbolic equations from text
and plotting them using the Mistral Document Library for file storage and retrieval.
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
mcp = FastMCP("Math Equation Plotter")

# Global storage for extracted equations (storing string expressions and SymPy objects)
extracted_equations: Dict[str, Dict[str, Any]] = {}

# Global storage for uploaded files
uploaded_files: Dict[str, Dict[str, Any]] = {}


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
    
    # Only add * before parentheses if it's not a function call
    # This is tricky - we'll be more conservative and only handle obvious cases
    # expr_str = re.sub(r'([a-zA-Z_])(\()', r'\1*\2', expr_str)  # Removed this line
    
    return expr_str


def plot_equations_to_mistral(equation_ids: List[str], 
                             x_range: tuple = (-10, 10), 
                             y_range: Optional[tuple] = None,
                             resolution: int = 1000,
                             title: str = "Mathematical Functions") -> Dict[str, Any]:
    """
    Plot one or more extracted equations and upload to Mistral Document Library.
    
    Args:
        equation_ids (List[str]): List of equation IDs to plot
        x_range (tuple): X-axis range (min, max)
        y_range (Optional[tuple]): Y-axis range (min, max), auto if None
        resolution (int): Number of points for plotting
        title (str): Plot title
        
    Returns:
        Dict[str, Any]: Upload result with file information
        
    Examples:
        >>> plot_equations_to_mistral(['eq1'], x_range=(-5, 5))
        {'success': True, 'file_id': 'file_123', 'file_name': 'plot.png', 'signed_url': 'https://...'}
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
        # Upload to Mistral Document Library
        from mistralai import Mistral
        
        # Check for API key
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        client = Mistral(api_key=api_key)
        
        # Generate filename
        filename = f"math_plot_{len(uploaded_files) + 1}.png"
        
        # Upload file
        uploaded_file = client.files.upload(
            file={
                "file_name": filename,
                "content": open(temp_path, "rb"),
            },
            purpose="ocr"  # Using OCR purpose for image processing
        )
        
        # Get signed URL
        signed_url_response = client.files.get_signed_url(file_id=uploaded_file.id)
        
        # Store file information
        file_info = {
            'file_id': uploaded_file.id,
            'file_name': filename,
            'signed_url': signed_url_response.url,
            'title': title,
            'equation_ids': equation_ids,
            'upload_time': uploaded_file.created_at
        }
        uploaded_files[uploaded_file.id] = file_info
        
        return {
            'success': True,
            'file_id': uploaded_file.id,
            'file_name': filename,
            'signed_url': signed_url_response.url,
            'title': title,
            'equation_ids': equation_ids,
            'message': f"Successfully uploaded plot to Mistral Document Library"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to upload plot to Mistral Document Library"
        }
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


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
    title="Plot Mathematical Equations to Mistral",
    description="Generate high-quality plots of extracted mathematical equations and upload them to the Mistral Document Library. Returns file information for retrieval and OCR processing."
)
def plot_extracted_equations(equation_ids: List[str],
                           x_range: List[float] = [-10, 10],
                           y_range: Optional[List[float]] = None,
                           resolution: int = 1000,
                           title: str = "Mathematical Functions") -> Dict[str, Any]:
    """
    Plot extracted equations and upload to Mistral Document Library.
    
    Args:
        equation_ids (List[str]): List of equation IDs to plot
        x_range (List[float]): X-axis range [min, max]
        y_range (Optional[List[float]]): Y-axis range [min, max]
        resolution (int): Number of points for plotting
        title (str): Plot title
        
    Returns:
        Dict[str, Any]: Upload result with file information
    """
    try:
        x_range_tuple = tuple(x_range)
        y_range_tuple = tuple(y_range) if y_range else None
        
        result = plot_equations_to_mistral(
            equation_ids=equation_ids,
            x_range=x_range_tuple,
            y_range=y_range_tuple,
            resolution=resolution,
            title=title
        )
        
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to plot and upload equations"
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
    title="List Uploaded Files",
    description="List all files uploaded to the Mistral Document Library, including their file IDs, names, and signed URLs for retrieval."
)
def list_uploaded_files() -> Dict[str, Any]:
    """
    List all uploaded files in Mistral Document Library.
    
    Returns:
        Dict[str, Any]: List of uploaded files
    """
    files_info = []
    for file_id, file_data in uploaded_files.items():
        files_info.append({
            "file_id": file_id,
            "file_name": file_data['file_name'],
            "title": file_data['title'],
            "equation_ids": file_data['equation_ids'],
            "signed_url": file_data['signed_url'],
            "upload_time": file_data['upload_time']
        })
    
    return {
        "success": True,
        "files": files_info,
        "count": len(files_info),
        "message": f"Found {len(files_info)} uploaded file(s)"
    }


@mcp.tool(
    title="Get File Signed URL",
    description="Get a signed URL for a specific file in the Mistral Document Library. Use this to retrieve or process the file."
)
def get_file_signed_url(file_id: str) -> Dict[str, Any]:
    """
    Get signed URL for a specific file.
    
    Args:
        file_id (str): The file ID from Mistral Document Library
        
    Returns:
        Dict[str, Any]: Signed URL information
    """
    try:
        if file_id not in uploaded_files:
            return {
                "success": False,
                "error": f"File {file_id} not found in uploaded files",
                "message": "File not found"
            }
        
        from mistralai import Mistral
        
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        client = Mistral(api_key=api_key)
        signed_url_response = client.files.get_signed_url(file_id=file_id)
        
        return {
            "success": True,
            "file_id": file_id,
            "signed_url": signed_url_response.url,
            "file_name": uploaded_files[file_id]['file_name'],
            "message": "Successfully retrieved signed URL"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get signed URL"
        }


@mcp.tool(
    title="Process File with OCR",
    description="Process an uploaded mathematical plot with Mistral's OCR to extract text and mathematical content from the image."
)
def process_file_with_ocr(file_id: str, include_image_base64: bool = True) -> Dict[str, Any]:
    """
    Process a file with Mistral OCR.
    
    Args:
        file_id (str): The file ID from Mistral Document Library
        include_image_base64 (bool): Whether to include base64 image in response
        
    Returns:
        Dict[str, Any]: OCR processing results
    """
    try:
        if file_id not in uploaded_files:
            return {
                "success": False,
                "error": f"File {file_id} not found in uploaded files",
                "message": "File not found"
            }
        
        from mistralai import Mistral
        
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        client = Mistral(api_key=api_key)
        
        # Get signed URL
        signed_url_response = client.files.get_signed_url(file_id=file_id)
        
        # Process with OCR
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url_response.url,
            },
            include_image_base64=include_image_base64
        )
        
        return {
            "success": True,
            "file_id": file_id,
            "file_name": uploaded_files[file_id]['file_name'],
            "ocr_results": ocr_response,
            "message": "Successfully processed file with OCR"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to process file with OCR"
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


@mcp.tool(
    title="Clear Uploaded Files Cache",
    description="Clear the local cache of uploaded files information. Note: This does not delete files from Mistral Document Library."
)
def clear_uploaded_files_cache() -> Dict[str, Any]:
    """
    Clear uploaded files cache from memory.
    
    Returns:
        Dict[str, Any]: Confirmation message
    """
    global uploaded_files
    count = len(uploaded_files)
    uploaded_files.clear()
    
    return {
        "success": True,
        "message": f"Cleared {count} uploaded file(s) from cache"
    }

