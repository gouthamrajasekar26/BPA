"""
MODULE 6: 3D Spatial Scene Reconstruction Render Engine
======================================================
Renders CyberLife "Detroit: Become Human" style 3D/4D Holographic Crime Scene 
Reconstructions featuring WebGL Three.js animated 4D time-lapse trajectory timelines,
fluid particle kinematics, impact shockwave flashes, and interactive viewports.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List, Dict, Tuple, Any
import plotly.graph_objects as go


def render_3d_matplotlib_scene(
    droplets: List[Dict[str, Any]],
    origin_3d: List[float],
    room_bounds: Tuple[float, float, float] = (800.0, 600.0, 300.0),
    save_path: str = "spatial_scene_3d.png"
) -> str:
    """Renders static 3D Matplotlib spatial scene blueprint with room wireframes."""
    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0B0F19')
    fig.patch.set_facecolor('#0B0F19')
    
    x_orig, y_orig, z_orig = origin_3d
    max_x, max_y, max_z = room_bounds
    
    # Grid Floor at Z=0
    gx, gy = np.meshgrid(np.linspace(0, max_x, 8), np.linspace(0, max_y, 8))
    ax.plot_wireframe(gx, gy, np.zeros_like(gx), color='#00F0FF', alpha=0.15, linewidth=0.8)
    
    # Wall Droplets
    drop_xs = [d["centroid"][0] for d in droplets]
    drop_ys = [d["centroid"][1] for d in droplets]
    drop_zs = [0.0] * len(droplets)
    ax.scatter(drop_xs, drop_ys, drop_zs, color='#00FF66', s=35, label='Wall Droplets (Z=0)', zorder=5)
    
    # Virtual Laser Stringing Vectors
    for idx, d in enumerate(droplets):
        dx, dy = d["centroid"]
        ax.plot([dx, x_orig], [dy, y_orig], [0.0, z_orig], color='#FF0055', linestyle='--', linewidth=1.2, alpha=0.7)

    # 3D Point of Origin
    ax.scatter([x_orig], [y_orig], [z_orig], color='#FF0055', s=280, marker='o', label='3D Point of Origin', zorder=10)
    ax.scatter([x_orig], [y_orig], [0.0], color='#00F0FF', s=90, marker='x', label='2D Convergence Focal Point', zorder=8)
    ax.plot([x_orig, x_orig], [y_orig, y_orig], [0.0, z_orig], color='#00F0FF', linestyle=':', linewidth=1.5)

    # Styling
    ax.set_title("CyberLife 3D Spatial Trajectory Blueprint", color='#00F0FF', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Wall X (cm)", color='#94A3B8', labelpad=8)
    ax.set_ylabel("Wall Y (cm)", color='#94A3B8', labelpad=8)
    ax.set_zlabel("Depth Z (cm)", color='#94A3B8', labelpad=8)
    ax.tick_params(colors='#64748B')
    
    ax.legend(loc='upper right', facecolor='#0F172A', edgecolor='#334155', labelcolor='white')
    ax.view_init(elev=22, azim=-50)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return save_path


def render_3d_plotly_interactive(
    droplets: List[Dict[str, Any]],
    origin_3d: List[float],
    room_bounds: Tuple[float, float, float] = (800.0, 600.0, 400.0)
) -> go.Figure:
    """
    Renders CyberLife Cyber-Matrix 3D Viewport in Plotly with room wireframes,
    neon laser trajectories, and glowing origin target reticles.
    """
    x_orig, y_orig, z_orig = origin_3d
    max_x, max_y, max_z = room_bounds

    fig = go.Figure()

    # Floor Grid
    for gx in np.linspace(0, max_x, 9):
        fig.add_trace(go.Scatter3d(x=[gx, gx], y=[0, max_y], z=[0, 0], mode='lines', line=dict(color='#00F0FF', width=1), opacity=0.25, showlegend=False))
    for gy in np.linspace(0, max_y, 9):
        fig.add_trace(go.Scatter3d(x=[0, max_x], y=[gy, gy], z=[0, 0], mode='lines', line=dict(color='#00F0FF', width=1), opacity=0.25, showlegend=False))

    # Room Bounding Box Wireframe
    box_lines = [
        ([0, max_x, max_x, 0, 0], [0, 0, max_y, max_y, 0], [0, 0, 0, 0, 0]),
        ([0, max_x, max_x, 0, 0], [0, 0, max_y, max_y, 0], [max_z, max_z, max_z, max_z, max_z]),
        ([0, 0], [0, 0], [0, max_z]), ([max_x, max_x], [0, 0], [0, max_z]),
        ([max_x, max_x], [max_y, max_y], [0, max_z]), ([0, 0], [max_y, max_y], [0, max_z])
    ]
    for lx, ly, lz in box_lines:
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='#3B82F6', width=2), opacity=0.4, showlegend=False))

    # Wall Droplets (Z=0)
    drop_xs = [d["centroid"][0] for d in droplets]
    drop_ys = [d["centroid"][1] for d in droplets]
    drop_zs = [0.0] * len(droplets)
    hover_texts = [f"<b>CYBERLIFE DROPLET #{d.get('id', i+1)}</b><br>Centroid: ({d['centroid'][0]:.1f}, {d['centroid'][1]:.1f})<br>Impact Angle: {d['impact_angle_deg']:.1f}°" for i, d in enumerate(droplets)]

    fig.add_trace(go.Scatter3d(
        x=drop_xs, y=drop_ys, z=drop_zs,
        mode='markers',
        marker=dict(size=7, color='#00FF66', symbol='circle', line=dict(color='#00F0FF', width=1)),
        name='Detected Droplets (Wall Z=0)',
        text=hover_texts,
        hoverinfo='text'
    ))

    # Neon Laser Stringing Trajectories
    for d in droplets:
        dx, dy = d["centroid"]
        fig.add_trace(go.Scatter3d(
            x=[dx, x_orig], y=[dy, y_orig], z=[0.0, z_orig],
            mode='lines',
            line=dict(color='#FF0055', width=4),
            opacity=0.75,
            showlegend=False,
            hoverinfo='none'
        ))

    # 3D Point of Origin Core
    fig.add_trace(go.Scatter3d(
        x=[x_orig], y=[y_orig], z=[z_orig],
        mode='markers+text',
        marker=dict(size=16, color='#FF0055', symbol='diamond', line=dict(color='#00F0FF', width=2)),
        name='3D Point of Origin Core',
        text=[f"ORIGIN: [{x_orig:.1f}, {y_orig:.1f}, {z_orig:.1f}] cm"],
        textposition="top center"
    ))

    # 2D Area of Convergence Projection
    fig.add_trace(go.Scatter3d(
        x=[x_orig], y=[y_orig], z=[0.0],
        mode='markers',
        marker=dict(size=10, color='#00F0FF', symbol='x'),
        name='2D Convergence Focal Point'
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_orig, x_orig], y=[y_orig, y_orig], z=[0.0, z_orig],
        mode='lines',
        line=dict(color='#00F0FF', width=2, dash='dot'),
        showlegend=False
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#070B12',
        plot_bgcolor='#070B12',
        scene=dict(
            xaxis=dict(title='Wall Width X (cm)', gridcolor='#1E293B', zerolinecolor='#00F0FF', backgroundcolor='#070B12'),
            yaxis=dict(title='Wall Height Y (cm)', gridcolor='#1E293B', zerolinecolor='#00F0FF', backgroundcolor='#070B12'),
            zaxis=dict(title='Outward Depth Z (cm)', gridcolor='#1E293B', zerolinecolor='#00F0FF', backgroundcolor='#070B12'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.4, y=-1.4, z=1.1))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=dict(text="CYBERLIFE DETECTIVE HUD: 3D CYBER-MATRIX VIEWPORT", font=dict(color='#00F0FF', size=16), x=0.5)
    )

    return fig


def generate_threejs_hologram_html(
    droplets: List[Dict[str, Any]],
    origin_3d: List[float],
    height: int = 650
) -> str:
    """
    Generates a CyberLife "Detroit: Become Human" 4D Time-Lapse Reconstruction Simulator 
    in Three.js WebGL. Features an interactive timeline slider (0.0s to 1.0s), Play/Pause 
    loop controls, fluid particle kinematics, impact shockwave flashes, and dynamic 
    wall spatter collision animations.
    """
    x_orig, y_orig, z_orig = origin_3d

    js_droplets = []
    for d in droplets:
        js_droplets.append({
            "id": d.get("id", 1),
            "x": float(d["centroid"][0]),
            "y": float(d["centroid"][1]),
            "z": 0.0,
            "angle": float(d.get("impact_angle_deg", 30.0))
        })

    droplets_json = json.dumps(js_droplets)

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: #050811;
                font-family: 'Segoe UI', Roboto, monospace, sans-serif;
                user-select: none;
            }}
            #canvas-container {{
                width: 100vw;
                height: {height}px;
                position: relative;
            }}
            
            /* CyberLife HUD Top Banner */
            .hud-overlay {{
                position: absolute;
                top: 14px;
                left: 16px;
                pointer-events: none;
                z-index: 10;
            }}
            .hud-badge {{
                background: rgba(0, 240, 255, 0.12);
                border: 1px solid #00F0FF;
                border-radius: 4px;
                color: #00F0FF;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .hud-dot {{
                width: 8px;
                height: 8px;
                background-color: #00F0FF;
                border-radius: 50%;
                box-shadow: 0 0 10px #00F0FF;
                animation: pulse 1.2s infinite alternate;
            }}
            @keyframes pulse {{
                0% {{ opacity: 0.3; transform: scale(0.8); }}
                100% {{ opacity: 1.0; transform: scale(1.3); }}
            }}

            /* Telemetry Panel */
            .hud-origin-coords {{
                position: absolute;
                top: 14px;
                right: 16px;
                background: rgba(15, 23, 42, 0.88);
                border: 1px solid #3B82F6;
                color: #E2E8F0;
                padding: 10px 14px;
                border-radius: 6px;
                font-size: 11px;
                font-family: monospace;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
                z-index: 10;
            }}

            /* 4D TIME-LAPSE CONTROLLER HUD BAR */
            .timeline-hud-bar {{
                position: absolute;
                bottom: 16px;
                left: 50%;
                transform: translateX(-50%);
                width: 90%;
                max-width: 850px;
                background: rgba(9, 13, 22, 0.92);
                border: 1px solid #00F0FF;
                border-radius: 10px;
                padding: 12px 20px;
                box-shadow: 0 0 25px rgba(0, 240, 255, 0.3);
                z-index: 20;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }}
            .timeline-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                color: #00F0FF;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            .timeline-controls {{
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            .timeline-btn {{
                background: rgba(0, 240, 255, 0.15);
                border: 1px solid #00F0FF;
                color: #00F0FF;
                padding: 6px 14px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            .timeline-btn:hover {{
                background: #00F0FF;
                color: #050811;
                box-shadow: 0 0 12px #00F0FF;
            }}
            .timeline-slider {{
                flex-grow: 1;
                -webkit-appearance: none;
                height: 6px;
                border-radius: 3px;
                background: #1e293b;
                outline: none;
            }}
            .timeline-slider::-webkit-slider-thumb {{
                -webkit-appearance: none;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #FF0055;
                border: 2px solid #00F0FF;
                cursor: pointer;
                box-shadow: 0 0 10px #FF0055;
            }}
            .speed-btn {{
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid #334155;
                color: #94A3B8;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 10px;
                cursor: pointer;
            }}
            .speed-btn.active {{
                border-color: #00F0FF;
                color: #00F0FF;
                background: rgba(0, 240, 255, 0.2);
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div class="hud-overlay">
                <div class="hud-badge">
                    <div class="hud-dot"></div>
                    CYBERLIFE 4D RECONSTRUCTION // SIMULATOR ACTIVE
                </div>
            </div>
            
            <div class="hud-origin-coords">
                <span style="color:#00F0FF;">SOLVED 3D ORIGIN:</span><br>
                X: {x_orig:.2f} cm | Y: {y_orig:.2f} cm | Z: {z_orig:.2f} cm<br>
                <span id="telemetry-status" style="color:#FF0055;">STATUS: 4D TRAJECTORY TIMELINE ACTIVE</span>
            </div>

            <!-- 4D TIME-LAPSE RECONSTRUCTION HUD BAR -->
            <div class="timeline-hud-bar">
                <div class="timeline-header">
                    <span>⏱️ 4D TRAJECTORY TIMELINE SCRUBBER</span>
                    <span id="time-display">FLIGHT TIME T = 0.00s / 1.00s</span>
                </div>
                <div class="timeline-controls">
                    <button class="timeline-btn" id="play-btn">▶ PLAY</button>
                    <button class="timeline-btn" id="reset-btn">⏪ REWIND</button>
                    <input type="range" min="0" max="100" value="0" class="timeline-slider" id="time-slider">
                    <button class="speed-btn active" data-speed="1.0">1.0x</button>
                    <button class="speed-btn" data-speed="0.5">0.5x</button>
                    <button class="speed-btn" data-speed="0.25">0.25x (Slow-Mo)</button>
                </div>
            </div>
        </div>

        <script>
            const dropletsData = {droplets_json};
            const originPos = {{ x: {x_orig}, y: {y_orig}, z: {z_orig} }};

            // Three.js Setup
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x050811, 0.0007);

            const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 5000);
            camera.position.set({x_orig + 420}, {y_orig - 320}, {z_orig + 420});

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setClearColor(0x050811, 1);
            container.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(originPos.x, originPos.y, originPos.z / 2);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Lighting
            scene.add(new THREE.AmbientLight(0x00f0ff, 0.6));
            const originLight = new THREE.PointLight(0xff0055, 3.0, 1500);
            originLight.position.set(originPos.x, originPos.y, originPos.z);
            scene.add(originLight);

            // Grid Floor (Z=0)
            const gridHelper = new THREE.GridHelper(1200, 30, 0x00f0ff, 0x1e293b);
            gridHelper.rotation.x = Math.PI / 2;
            gridHelper.position.set(400, 300, 0);
            scene.add(gridHelper);

            // 4D Trajectory Particle Objects & Splatters
            const flightParticles = [];
            const wallSplatters = [];

            // Origin Core Mesh & Shockwave
            const originGroup = new THREE.Group();
            originGroup.position.set(originPos.x, originPos.y, originPos.z);
            scene.add(originGroup);

            const coreMesh = new THREE.Mesh(
                new THREE.OctahedronGeometry(18, 2),
                new THREE.MeshStandardMaterial({{ color: 0xff0055, emissive: 0xff0055, wireframe: true }})
            );
            originGroup.add(coreMesh);

            // Origin Shockwave Ring
            const shockwaveGeo = new THREE.RingGeometry(2, 6, 32);
            const shockwaveMat = new THREE.MeshBasicMaterial({{ color: 0xff0055, side: THREE.DoubleSide, transparent: true, opacity: 0 }});
            const shockwaveMesh = new THREE.Mesh(shockwaveGeo, shockwaveMat);
            originGroup.add(shockwaveMesh);

            // Create Droplet Particles and Laser Paths
            dropletsData.forEach(drop => {{
                // Laser Stringing Ray Line
                const points = [
                    new THREE.Vector3(drop.x, drop.y, 0),
                    new THREE.Vector3(originPos.x, originPos.y, originPos.z)
                ];
                const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
                const lineMat = new THREE.LineDashedMaterial({{ color: 0xff0055, linewidth: 2, scale: 1, dashSize: 8, gapSize: 4 }});
                const line = new THREE.Line(lineGeo, lineMat);
                line.computeLineDistances();
                scene.add(line);

                // Traveling 4D Blood Droplet Mesh
                const dropMat = new THREE.MeshBasicMaterial({{ color: 0xff0055 }});
                const dropMesh = new THREE.Mesh(new THREE.SphereGeometry(4, 16, 16), dropMat);
                dropMesh.position.set(originPos.x, originPos.y, originPos.z);
                scene.add(dropMesh);

                // Wall Collision Splatter Footprint Ring
                const splatMat = new THREE.MeshBasicMaterial({{ color: 0x00ff66, side: THREE.DoubleSide, transparent: true, opacity: 0 }});
                const splatMesh = new THREE.Mesh(new THREE.RingGeometry(1, 8, 16), splatMat);
                splatMesh.position.set(drop.x, drop.y, 0.5);
                scene.add(splatMesh);

                flightParticles.push({{
                    mesh: dropMesh,
                    start: new THREE.Vector3(originPos.x, originPos.y, originPos.z),
                    end: new THREE.Vector3(drop.x, drop.y, 0),
                    splatMesh: splatMesh
                }});
            }});

            // 4D Timeline State Management
            let progress = 0.0; // 0.0 to 1.0
            let isPlaying = true;
            let speedMultiplier = 1.0;

            const timeSlider = document.getElementById('time-slider');
            const playBtn = document.getElementById('play-btn');
            const resetBtn = document.getElementById('reset-btn');
            const timeDisplay = document.getElementById('time-display');
            const telemetryStatus = document.getElementById('telemetry-status');

            playBtn.addEventListener('click', () => {{
                isPlaying = !isPlaying;
                playBtn.innerText = isPlaying ? "⏸ PAUSE" : "▶ PLAY";
            }});

            resetBtn.addEventListener('click', () => {{
                progress = 0.0;
                timeSlider.value = 0;
            }});

            timeSlider.addEventListener('input', (e) => {{
                progress = parseFloat(e.target.value) / 100.0;
                isPlaying = false;
                playBtn.innerText = "▶ PLAY";
            }});

            document.querySelectorAll('.speed-btn').forEach(btn => {{
                btn.addEventListener('click', (e) => {{
                    document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    speedMultiplier = parseFloat(e.target.getAttribute('data-speed'));
                }});
            }});

            // 60FPS Render & 4D Physics Loop
            let clock = new THREE.Clock();

            function animate() {{
                requestAnimationFrame(animate);

                const delta = clock.getDelta();

                if (isPlaying) {{
                    progress += (delta * 0.4) * speedMultiplier;
                    if (progress > 1.0) progress = 0.0;
                    timeSlider.value = progress * 100;
                }}

                // Update UI Telemetry
                timeDisplay.innerText = `FLIGHT TIME T = ${{ (progress * 1.0).toFixed(2) }}s / 1.00s`;
                
                if (progress < 0.1) {{
                    telemetryStatus.innerText = "STATUS: 🔴 WEAPON DISCHARGE / IMPACT TRIGGER";
                    telemetryStatus.style.color = "#FF0055";
                    // Shockwave animation
                    shockwaveMesh.scale.set(1 + progress * 20, 1 + progress * 20, 1);
                    shockwaveMat.opacity = 1.0 - progress * 10;
                }} else if (progress < 0.95) {{
                    telemetryStatus.innerText = "STATUS: ✈️ DROPLET FLIGHT TRAJECTORY VECTOR";
                    telemetryStatus.style.color = "#00F0FF";
                    shockwaveMat.opacity = 0;
                }} else {{
                    telemetryStatus.innerText = "STATUS: 💥 WALL PLANE COLLISION & SPATTER SPLAT";
                    telemetryStatus.style.color = "#00FF66";
                }}

                // Update 4D Particle Positions
                flightParticles.forEach(p => {{
                    // Interpolate position from Origin (t=0) to Wall (t=1)
                    p.mesh.position.lerpVectors(p.start, p.end, progress);

                    // Dynamic Splatter Footprint Opacity on Wall Impact
                    if (progress > 0.9) {{
                        p.splatMesh.material.opacity = (progress - 0.9) * 10.0;
                        p.splatMesh.scale.set(1 + (progress - 0.9) * 5, 1 + (progress - 0.9) * 5, 1);
                    }} else {{
                        p.splatMesh.material.opacity = 0;
                    }}
                }});

                coreMesh.rotation.y += 0.01;
                coreMesh.rotation.x += 0.005;

                controls.update();
                renderer.render(scene, camera);
            }}

            animate();

            window.addEventListener('resize', () => {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html_code


if __name__ == "__main__":
    print("=== MODULE 6: CyberLife 4D Time-Lapse Reconstruction Render ===")
    sample_drops = [{"id": 1, "centroid": (200, 300), "impact_angle_deg": 35.0}]
    sample_origin = [350.0, 350.0, 150.0]
    render_3d_matplotlib_scene(sample_drops, sample_origin)
    print("[Module 6] Execution complete!\n")
