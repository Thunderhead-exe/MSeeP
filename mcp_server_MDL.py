"""
Model Context Protocol (MCP) Server for Mathematical Equation Plotting - MDL IMG 2 Version

This server provides tools for extracting symbolic equations from text
and plotting them using the Mistral Document Library for image storage and retrieval.
This version uploads images to the library and retrieves them back.
"""

import os
import re
import base64
import io
import tempfile
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Math Equation Plotter MDL IMG 2")

# Global storage for extracted equations (storing string expressions and SymPy objects)
extracted_equations: Dict[str, Dict[str, Any]] = {}

# Global storage for uploaded files and libraries
uploaded_files: Dict[str, Dict[str, Any]] = {}
mseep_library_id: str = None  # Store the MSeeP_Plots library ID


def get_or_create_mseep_library() -> str:
    """
    Get or create the MSeeP_Plots library for storing equation plots.
    
    Returns:
        str: The library ID
    """
    global mseep_library_id
    
    if mseep_library_id:
        return mseep_library_id
    
    try:
        from mistralai import Mistral
        from mistralai.models import File
        
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        client = Mistral(api_key=api_key)
        
        # List libraries to check if MSeeP_Plots exists
        libraries = client.beta.libraries.list().data
        for library in libraries:
            if library.name == "MSeeP_Plots":
                mseep_library_id = library.id
                return mseep_library_id
        
        # Create new library if not found
        new_library = client.beta.libraries.create(
            name="MSeeP_Plots",
            description="A library for storing mathematical equation plots and visualizations from MSeeP"
        )
        mseep_library_id = new_library.id
        return mseep_library_id
        
    except Exception as e:
        raise Exception(f"Failed to create/get MSeeP_Plots library: {str(e)}")


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


def plot_equations_to_mdl_library(
    equation_ids: List[str],
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    title: Optional[str] = None,
    resolution: int = 1000
) -> Dict[str, Any]:
    """
    Plot extracted equations, save to temporary PNG file, upload to Mistral Document Library, and retrieve the image.
    
    Args:
        equation_ids (List[str]): List of equation IDs to plot
        x_range (Optional[List[float]]): X-axis range [min, max]
        y_range (Optional[List[float]]): Y-axis range [min, max] (for implicit equations)
        title (Optional[str]): Plot title
        resolution (int): Plot resolution
        
    Returns:
        Dict[str, Any]: Plot result with image data from library
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
        from mistralai.models import File
        
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        client = Mistral(api_key=api_key)
        
        # Get or create MSeeP_Plots library
        library_id = get_or_create_mseep_library()
        
        # Generate filename
        filename = f"mseep_plot_{len(uploaded_files) + 1}.png"
        
        # Upload document to Mistral Document Library
        with open(temp_path, "rb") as file_content:
            uploaded_doc = client.beta.libraries.documents.upload(
                library_id=library_id,
                file=File(fileName=filename, content=file_content)
            )
        
        # Check status and wait for processing to finish
        status = client.beta.libraries.documents.status(
            library_id=library_id, 
            document_id=uploaded_doc.id
        )
        
        import time
        while status.processing_status == "Running":
            status = client.beta.libraries.documents.status(
                library_id=library_id, 
                document_id=uploaded_doc.id
            )
            time.sleep(1)
        
        # Get document info once processed
        final_doc = client.beta.libraries.documents.get(
            library_id=library_id, 
            document_id=uploaded_doc.id
        )
        
        # Read the temporary file as PNG binary data
        with open(temp_path, "rb") as image_file:
            png_image_data = image_file.read()
        
        # Store file information
        file_info = {
            'document_id': uploaded_doc.id,
            'library_id': library_id,
            'filename': filename,
            'uploaded_at': datetime.now().isoformat(),
            'equation_ids': equation_ids,
            'title': title,
            'x_range': x_range,
            'y_range': y_range,
            'processing_status': final_doc.processing_status,
            'summary': final_doc.summary
        }
        uploaded_files[uploaded_doc.id] = file_info
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return {
            'success': True,
            'document_id': uploaded_doc.id,
            'library_id': library_id,
            'filename': filename,
            'processing_status': final_doc.processing_status,
            'summary': final_doc.summary,
            'title': title,
            'equation_ids': equation_ids,
            'image_png': png_image_data,
            'image_format': 'png',
            'image_size': len(png_image_data),
            'message': f"Successfully uploaded plot to MSeeP_Plots library and retrieved PNG image data"
        }
        
    except Exception as e:
        # Clean up temporary file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to upload plot to Mistral Document Library"
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
    title="Plot Extracted Equations to MDL Library",
    description="Plot one or more extracted mathematical equations, save as temporary PNG file, upload to MSeeP_Plots library, and retrieve the PNG image data. Returns PNG binary image data from the library."
)
def plot_extracted_equations(
    equation_ids: List[str],
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Plot extracted equations and get image from library.
    
    Args:
        equation_ids (List[str]): List of equation IDs to plot
        x_range (Optional[List[float]]): X-axis range [min, max]
        y_range (Optional[List[float]]): Y-axis range [min, max]
        title (Optional[str]): Plot title
        
    Returns:
        Dict[str, Any]: Plot result with image data from library
    """
    return plot_equations_to_mdl_library(equation_ids, x_range, y_range, title)


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
    title="List Uploaded Files",
    description="List all uploaded files in the MSeeP_Plots library with their document IDs and metadata."
)
def list_uploaded_files() -> Dict[str, Any]:
    """
    List all uploaded files in MSeeP_Plots library.
    
    Returns:
        Dict[str, Any]: List of uploaded files
    """
    files_info = []
    for document_id, file_data in uploaded_files.items():
        files_info.append({
            "document_id": document_id,
            "library_id": file_data.get('library_id', 'unknown'),
            "filename": file_data.get('filename', 'unknown'),
            "title": file_data.get('title', 'unknown'),
            "equation_ids": file_data.get('equation_ids', []),
            "processing_status": file_data.get('processing_status', 'unknown'),
            "summary": file_data.get('summary', ''),
            "uploaded_at": file_data.get('uploaded_at', 'unknown')
        })
    
    return {
        "success": True,
        "files": files_info,
        "count": len(files_info),
        "message": f"Found {len(files_info)} uploaded file(s)"
    }


@mcp.tool(
    title="Get Document Information",
    description="Get detailed information about a specific uploaded document from the MSeeP_Plots library."
)
def get_document_info(document_id: str) -> Dict[str, Any]:
    """
    Get detailed information for a specific document.
    
    Args:
        document_id (str): The document ID from MSeeP_Plots library
        
    Returns:
        Dict[str, Any]: Document information
    """
    try:
        if document_id not in uploaded_files:
            return {
                "success": False,
                "error": f"Document {document_id} not found in uploaded files",
                "message": "Document not found"
            }
        
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("mistralai package not installed. Install with: pip install mistralai")
        
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        client = Mistral(api_key=api_key)
        file_info = uploaded_files[document_id]
        library_id = file_info['library_id']
        
        # Get document details
        document = client.beta.libraries.documents.get(
            library_id=library_id,
            document_id=document_id
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "library_id": library_id,
            "filename": document.name,
            "processing_status": document.processing_status,
            "summary": document.summary,
            "size": document.size,
            "mime_type": document.mime_type,
            "created_at": document.created_at,
            "message": "Document information retrieved successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get document information"
        }


@mcp.tool(
    title="Get Library Information",
    description="Get information about the MSeeP_Plots library and its documents."
)
def get_library_info() -> Dict[str, Any]:
    """
    Get information about the MSeeP_Plots library and its documents.
    
    Returns:
        Dict[str, Any]: Library information
    """
    try:
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("mistralai package not installed. Install with: pip install mistralai")
        
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        client = Mistral(api_key=api_key)
        
        # Get or create library
        library_id = get_or_create_mseep_library()
        
        # Get library details
        library = client.beta.libraries.get(library_id=library_id)
        
        # List documents in library
        documents = client.beta.libraries.documents.list(library_id=library_id)
        
        return {
            "success": True,
            "library_id": library_id,
            "library_name": library.name,
            "library_description": library.description,
            "total_documents": library.nb_documents,
            "total_size": library.total_size,
            "created_at": library.created_at,
            "updated_at": library.updated_at,
            "documents": [
                {
                    "document_id": doc.id,
                    "name": doc.name,
                    "size": doc.size,
                    "mime_type": doc.mime_type,
                    "processing_status": doc.processing_status,
                    "created_at": doc.created_at
                }
                for doc in documents.data
            ],
            "message": "Library information retrieved successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get library information"
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


@mcp.tool(
    title="Clear Uploaded Files Cache",
    description="Clear the local cache of uploaded files information. This does not delete files from the Mistral Document Library, only clears the local tracking."
)
def clear_uploaded_files_cache() -> Dict[str, Any]:
    """
    Clear uploaded files cache.
    
    Returns:
        Dict[str, Any]: Clear operation result
    """
    global uploaded_files
    count = len(uploaded_files)
    uploaded_files.clear()
    
    return {
        'success': True,
        'count': count,
        'message': f'Cleared {count} file(s) from cache'
    }
