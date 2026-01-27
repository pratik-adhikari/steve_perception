"""
Logging decorators for automatic function logging.
Reduces boilerplate and standardizes logging format.
"""
import functools
import logging
import time
from typing import Any, Callable


def log_function_call(func: Callable) -> Callable:
    """
    Decorator to automatically log function entry, exit, duration, and errors.
    
    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            return result
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger(func.__module__)
        func_name = f"{func.__qualname__}"
        
        # Log entry
        args_repr = [repr(a) for a in args[:2]]  # Limit to first 2 args to avoid huge logs
        kwargs_repr = [f"{k}={v!r}" for k, v in list(kwargs.items())[:2]]
        signature = ", ".join(args_repr + kwargs_repr)
        if len(args) > 2 or len(kwargs) > 2:
            signature += ", ..."
        
        logger.info(f"→ {func_name}({signature})")
        
        # Execute function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log successful exit
            logger.info(f"← {func_name} completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"✗ {func_name} failed after {duration:.2f}s: {e}", exc_info=True)
            raise
    
    return wrapper


def log_method_call(func: Callable) -> Callable:
    """
    Decorator for logging class methods (includes class name).
    
    Usage:
        class MyClass:
            @log_method_call
            def my_method(self, arg1):
                return result
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        logger = logging.getLogger(func.__module__)
        class_name = self.__class__.__name__
        func_name = f"{class_name}.{func.__name__}"
        
        # Log entry with simplified args
        args_repr = [repr(a) for a in args[:2]]
        kwargs_repr = [f"{k}={v!r}" for k, v in list(kwargs.items())[:2]]
        signature = ", ".join(args_repr + kwargs_repr)
        if len(args) > 2 or len(kwargs) > 2:
            signature += ", ..."
            
        logger.info(f"→ {func_name}({signature})")
        
        # Execute method
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"← {func_name} completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"✗ {func_name} failed after {duration:.2f}s: {e}", exc_info=True)
            raise
    
    return wrapper
