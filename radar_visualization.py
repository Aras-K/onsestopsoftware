# radar_visualization.py - Enhanced Professional Visualization System
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)

class EnhancedRadarVisualization:
    """Enhanced professional radar target visualization with numbered targets and detailed info."""

    # Color scheme for different target types
    COLORS = {
        'vessel': (0, 255, 0),      # Green
        'landmass': (0, 165, 255),  # Orange
        'obstacle': (0, 0, 255),    # Red
        'unknown': (128, 128, 128)  # Gray
    }

    # Risk level colors
    RISK_COLORS = {
        'CRITICAL': (0, 0, 255),    # Red
        'HIGH': (0, 69, 255),       # Orange-red
        'MEDIUM': (0, 140, 255),    # Orange
        'LOW': (0, 255, 255),       # Yellow
        'SAFE': (0, 255, 0)         # Green
    }

    @staticmethod
    def create_numbered_visualization(image_input, targets_metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """Create visualization with numbered targets on image and detailed info below."""
        # Handle both file paths and numpy arrays
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                return None, {}
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise ValueError("image_input must be either a file path (str) or numpy array")

        # Create a copy for annotation
        annotated = image.copy()

        # Add semi-transparent overlay for better visibility
        overlay = annotated.copy()

        targets = targets_metadata.get('targets', [])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 50

        # Draw targets with numbers
        for target in targets:
            target_id = target['id']
            range_nm = target['range_nm']
            bearing_deg = target['bearing_deg']
            target_type = target['type']
            risk_level = target.get('risk_level', 'LOW')
            confidence = target.get('confidence', 0.5)

            # Calculate position
            bearing_rad = np.radians(bearing_deg - 90)
            max_range = 12.0  # Default max range
            pixel_distance = (range_nm / max_range) * radius

            x = int(center[0] + pixel_distance * np.cos(bearing_rad))
            y = int(center[1] + pixel_distance * np.sin(bearing_rad))

            # Get colors based on type and risk
            type_color = EnhancedRadarVisualization.COLORS.get(target_type, (128, 128, 128))
            risk_color = EnhancedRadarVisualization.RISK_COLORS.get(risk_level, (128, 128, 128))

            # Draw target circle with risk-based color
            circle_color = risk_color if risk_level in ['CRITICAL', 'HIGH'] else type_color
            thickness = 3 if risk_level in ['CRITICAL', 'HIGH'] else 2
            radius_circle = 15 if risk_level == 'CRITICAL' else 12

            cv2.circle(overlay, (x, y), radius_circle, circle_color, thickness)

            # Draw crosshair for vessels
            if target_type == 'vessel':
                cv2.line(overlay, (x-18, y), (x+18, y), circle_color, 2)
                cv2.line(overlay, (x, y-18), (x, y+18), circle_color, 2)

            # Draw target number in the center
            cv2.circle(overlay, (x, y), 8, (255, 255, 255), -1)  # White background
            cv2.putText(overlay, str(target_id), (x-4, y+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Add range label
            label = f"{range_nm:.1f}NM"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness_text = 1

            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness_text)

            # Draw background rectangle
            cv2.rectangle(overlay,
                         (x + 12, y - text_height - 2),
                         (x + 12 + text_width + 4, y + 2),
                         (0, 0, 0), -1)

            # Draw text
            cv2.putText(overlay, label, (x + 14, y),
                       font, font_scale, circle_color, thickness_text)

        # Blend overlay with original
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

        # Add legend
        legend_info = EnhancedRadarVisualization._add_legend(annotated, targets_metadata)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, f"Generated: {timestamp}", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return annotated, legend_info

    @staticmethod
    def _add_legend(image: np.ndarray, targets_metadata: Dict) -> Dict:
        """Add comprehensive legend to the image."""
        h, w = image.shape[:2]

        # Legend background
        legend_x = w - 250
        legend_y_start = 30
        legend_height = 200
        legend_width = 240

        cv2.rectangle(image, (legend_x - 10, 10),
                     (legend_x + legend_width, legend_y_start + legend_height),
                     (0, 0, 0), -1)
        cv2.rectangle(image, (legend_x - 10, 10),
                     (legend_x + legend_width, legend_y_start + legend_height),
                     (255, 255, 255), 1)

        # Legend title
        cv2.putText(image, "TARGET DETECTION", (legend_x, legend_y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Target counts
        vessels = targets_metadata.get('vessels', 0)
        landmasses = targets_metadata.get('landmasses', 0)
        obstacles = targets_metadata.get('obstacles', 0)
        total = targets_metadata.get('total', 0)

        y_pos = legend_y_start + 45
        items = [
            (vessels, "Vessels", EnhancedRadarVisualization.COLORS['vessel']),
            (landmasses, "Landmasses", EnhancedRadarVisualization.COLORS['landmass']),
            (obstacles, "Obstacles", EnhancedRadarVisualization.COLORS['obstacle'])
        ]

        for count, name, color in items:
            if count > 0:
                cv2.circle(image, (legend_x + 10, y_pos), 5, color, -1)
                cv2.putText(image, f"{name}: {count}", (legend_x + 25, y_pos + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 25

        # Risk levels
        y_pos += 10
        cv2.putText(image, "RISK LEVELS:", (legend_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        risk_items = [
            ("CRITICAL", EnhancedRadarVisualization.RISK_COLORS['CRITICAL']),
            ("HIGH", EnhancedRadarVisualization.RISK_COLORS['HIGH']),
            ("MEDIUM", EnhancedRadarVisualization.RISK_COLORS['MEDIUM'])
        ]

        for risk, color in risk_items:
            y_pos += 20
            cv2.circle(image, (legend_x + 10, y_pos), 3, color, -1)
            cv2.putText(image, risk, (legend_x + 25, y_pos + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return {
            'total_targets': total,
            'vessels': vessels,
            'landmasses': landmasses,
            'obstacles': obstacles,
            'legend_position': (legend_x, legend_y_start, legend_width, legend_height)
        }

    @staticmethod
    def generate_target_details_table(targets_metadata: Dict) -> str:
        """Generate detailed HTML table of target information."""
        targets = targets_metadata.get('targets', [])

        if not targets:
            return "<p>No targets detected.</p>"

        # Sort targets by ID
        targets.sort(key=lambda x: x['id'])

        html = """
        <div style="margin: 20px 0;">
            <h3 style="color: #1f2937; margin-bottom: 15px;">🎯 Target Details</h3>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <thead>
                        <tr style="background-color: #f3f4f6;">
                            <th style="padding: 12px; text-align: left; border: 1px solid #e5e7eb; font-weight: 600;">ID</th>
                            <th style="padding: 12px; text-align: left; border: 1px solid #e5e7eb; font-weight: 600;">Type</th>
                            <th style="padding: 12px; text-align: left; border: 1px solid #e5e7eb; font-weight: 600;">Distance</th>
                            <th style="padding: 12px; text-align: left; border: 1px solid #e5e7eb; font-weight: 600;">Bearing</th>
                            <th style="padding: 12px; text-align: left; border: 1px solid #e5e7eb; font-weight: 600;">Risk Level</th>
                            <th style="padding: 12px; text-align: left; border: 1px solid #e5e7eb; font-weight: 600;">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for target in targets:
            target_id = target['id']
            target_type = target['type'].upper()
            distance_nm = target['range_nm']
            distance_m = target['distance_meters']
            bearing = target['bearing_deg']
            risk_level = target.get('risk_level', 'LOW')
            confidence = target.get('confidence', 0.5)

            # Color coding for risk levels
            risk_colors = {
                'CRITICAL': '#dc2626',
                'HIGH': '#ea580c',
                'MEDIUM': '#d97706',
                'LOW': '#65a30d',
                'SAFE': '#16a34a'
            }

            risk_color = risk_colors.get(risk_level, '#6b7280')

            # Type colors
            type_colors = {
                'VESSEL': '#10b981',
                'LANDMASS': '#f59e0b',
                'OBSTACLE': '#ef4444',
                'UNKNOWN': '#6b7280'
            }

            type_color = type_colors.get(target_type, '#6b7280')

            html += f"""
                        <tr style="border: 1px solid #e5e7eb;">
                            <td style="padding: 12px; border: 1px solid #e5e7eb; font-weight: 600; color: #1f2937;">{target_id}</td>
                            <td style="padding: 12px; border: 1px solid #e5e7eb;">
                                <span style="background-color: {type_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 500;">{target_type}</span>
                            </td>
                            <td style="padding: 12px; border: 1px solid #e5e7eb;">
                                <div style="font-weight: 600;">{distance_nm:.1f} NM</div>
                                <div style="font-size: 12px; color: #6b7280;">{distance_m:,.0f} meters</div>
                            </td>
                            <td style="padding: 12px; border: 1px solid #e5e7eb; font-weight: 600;">{bearing:.0f}°</td>
                            <td style="padding: 12px; border: 1px solid #e5e7eb;">
                                <span style="background-color: {risk_color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 500;">{risk_level}</span>
                            </td>
                            <td style="padding: 12px; border: 1px solid #e5e7eb;">
                                <div style="font-weight: 600;">{confidence:.1%}</div>
                            </td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """

        return html

    @staticmethod
    def generate_summary_stats(targets_metadata: Dict) -> Dict:
        """Generate summary statistics for the targets."""
        targets = targets_metadata.get('targets', [])

        if not targets:
            return {
                'total_targets': 0,
                'avg_distance_nm': 0,
                'closest_target': None,
                'farthest_target': None,
                'risk_distribution': {},
                'type_distribution': {}
            }

        distances = [t['range_nm'] for t in targets]
        risk_levels = [t.get('risk_level', 'LOW') for t in targets]
        types = [t['type'] for t in targets]

        # Calculate statistics
        avg_distance = sum(distances) / len(distances)
        closest = min(targets, key=lambda x: x['range_nm'])
        farthest = max(targets, key=lambda x: x['range_nm'])

        # Risk distribution
        risk_dist = {}
        for risk in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'SAFE']:
            risk_dist[risk] = risk_levels.count(risk)

        # Type distribution
        type_dist = {}
        for t in ['vessel', 'landmass', 'obstacle', 'unknown']:
            type_dist[t] = types.count(t)

        return {
            'total_targets': len(targets),
            'avg_distance_nm': round(avg_distance, 2),
            'closest_target': {
                'id': closest['id'],
                'distance_nm': closest['range_nm'],
                'type': closest['type']
            },
            'farthest_target': {
                'id': farthest['id'],
                'distance_nm': farthest['range_nm'],
                'type': farthest['type']
            },
            'risk_distribution': risk_dist,
            'type_distribution': type_dist
        }

    @staticmethod
    def save_visualization(image_path: str, targets_metadata: Dict,
                         output_path: str = None) -> str:
        """
        Save visualization to file.

        Args:
            image_path: Original image path
            targets_metadata: Detected targets metadata
            output_path: Where to save (optional)

        Returns:
            Path to saved visualization
        """
        import os

        # Create visualization
        viz_image, _ = EnhancedRadarVisualization.create_numbered_visualization(
            image_path, targets_metadata
        )

        if viz_image is None:
            return None

        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_targets_numbered.png"

        # Save
        cv2.imwrite(output_path, viz_image)
        logger.info(f"Saved enhanced visualization to: {output_path}")

        return output_path