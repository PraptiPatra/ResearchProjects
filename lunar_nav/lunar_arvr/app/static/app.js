(() => {
  // --------- DOM handles ----------
  const modal = document.getElementById('coordModal');
  const suggestions = document.getElementById('suggestions');
  const coordForm = document.getElementById('coordForm');
  const mapCanvas = document.getElementById('mapCanvas');
  const mapWrapper = document.getElementById('mapWrapper');
  const routesPanel = document.getElementById('routesPanel');
  const detailsPanel = document.getElementById('routeDetails');
  const vrButton = document.getElementById('startVr');
  const vrContainer = document.getElementById('vrContainer');
  const exitVrBtn = document.getElementById('exitVr');
  const vrScene = document.getElementById('vrScene');

  // --------- State ----------
  let currentTerrain = null;
  let currentRoutes = [];
  let selectedRoute = null;

  const ctx = mapCanvas.getContext('2d');

  // --------- Helpers ----------
  function demToImageData(dem) {
    const h = dem.length, w = dem[0].length;
    const off = document.createElement('canvas');
    off.width = w; off.height = h;
    const octx = off.getContext('2d');
    const imgData = octx.createImageData(w, h);

    // normalize DEM to 0..255
    let min = Infinity, max = -Infinity;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = dem[y][x];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const range = max - min + 1e-6;
    let idx = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const c = Math.floor(((dem[y][x] - min) / range) * 255);
        imgData.data[idx++] = c;
        imgData.data[idx++] = c;
        imgData.data[idx++] = c;
        imgData.data[idx++] = 255;
      }
    }
    octx.putImageData(imgData, 0, 0);
    return off;
  }

  function drawMap(dem, haz) {
    const img = demToImageData(dem);
    const aspect = img.width / img.height;

    let drawW = window.innerWidth;
    let drawH = drawW / aspect;
    if (drawH > window.innerHeight) {
      drawH = window.innerHeight;
      drawW = drawH * aspect;
    }

    mapCanvas.width = drawW;
    mapCanvas.height = drawH;

    ctx.clearRect(0, 0, drawW, drawH);
    ctx.drawImage(img, 0, 0, drawW, drawH);

    // Hazard heat overlay (red, semi-transparent)
    if (haz) {
      const hCanvas = document.createElement('canvas');
      hCanvas.width = haz[0].length;
      hCanvas.height = haz.length;
      const hctx = hCanvas.getContext('2d');
      const id = hctx.createImageData(hCanvas.width, hCanvas.height);
      let i = 0;
      for (let y = 0; y < hCanvas.height; y++) {
        for (let x = 0; x < hCanvas.width; x++) {
          const hv = haz[y][x];
          const a = Math.min(1, hv * 1.2); // boost slightly
          id.data[i++] = 255;       // R
          id.data[i++] = 0;         // G
          id.data[i++] = 0;         // B
          id.data[i++] = Math.floor(a * 255); // A
        }
      }
      hctx.putImageData(id, 0, 0);
      ctx.drawImage(hCanvas, 0, 0, drawW, drawH);
    }
  }

  function toCanvasCoords(path) {
    const w = currentTerrain.shape[1];
    const h = currentTerrain.shape[0];
    const cw = mapCanvas.width;
    const ch = mapCanvas.height;
    return path.map(pt => ({
      x: (pt.x / (w - 1)) * cw,
      y: (pt.y / (h - 1)) * ch
    }));
  }

  function drawRoutes() {
    // Redraw map
    drawMap(currentTerrain.dem, currentTerrain.hazards);

    const palette = { green: '#00ffae', yellow: '#ffe96a', red: '#ff6b6b' };

    currentRoutes.forEach(route => {
      const color = palette[route.color] || '#8888ff';
      const dim = (selectedRoute && selectedRoute !== route) ? 0.3 : 1.0;
      ctx.save();
      ctx.globalAlpha = dim;
      const pts = toCanvasCoords(route.path);
      ctx.strokeStyle = color;
      ctx.lineWidth = (selectedRoute === route) ? 3 : 1.5;
      ctx.beginPath();
      pts.forEach((p, i) => {
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
      ctx.restore();
    });
  }

  // --------- Data flow ----------
  async function loadSuggestions() {
    try {
      const res = await fetch('/images');
      const data = await res.json();
      const imgs = data.images || [];

      suggestions.innerHTML = '';
      imgs.slice(0, 5).forEach(img => {
        const chip = document.createElement('div');
        chip.className = 'suggestion';
        chip.textContent = `ID ${img.image_id}: (${img.suggest_x}, ${img.suggest_y})`;
        chip.onclick = () => {
          coordForm.imageId.value = img.image_id;
          coordForm.coordX.value = img.suggest_x;
          coordForm.coordY.value = img.suggest_y;
        };
        suggestions.appendChild(chip);
      });
    } catch (e) {
      console.error('Failed to load suggestions', e);
    }
  }

  async function computeRoutes(imageId, x, y) {
    try {
      const tRes = await fetch(`/terrain?image_id=${imageId}`);
      if (!tRes.ok) {
        alert('Terrain not found for that image. Make sure .npy files exist in lunar_arvr/data.');
        return;
      }
      currentTerrain = await tRes.json();

      drawMap(currentTerrain.dem, currentTerrain.hazards);

      const req = {
        image_id: imageId,
        landing_xy: [x, y],
        min_length_m: 100.0,
        meters_per_pixel: currentTerrain.meters_per_pixel || 1.0,
        top_k: 5
      };

      const rRes = await fetch('/routes', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(req)
      });

      if (!rRes.ok) {
        alert('Route computation failed on the server.');
        return;
      }

      const rData = await rRes.json();
      currentRoutes = rData.routes || [];
      selectedRoute = null;

      renderRoutesList();
      drawRoutes();
    } catch (err) {
      console.error('Error computing routes', err);
      alert('Route computation failed. Check console for details.');
    }
  }

  // --------- UI panels ----------
  function renderRoutesList() {
    routesPanel.innerHTML = '';
    routesPanel.classList.remove('hidden');
    detailsPanel.classList.add('hidden');
    vrButton.classList.add('hidden');

    const title = document.createElement('h3');
    title.textContent = 'Top Routes';
    routesPanel.appendChild(title);

    currentRoutes.forEach((route, idx) => {
      const item = document.createElement('div');
      item.className = 'route-item';
      item.innerHTML = `
        <strong>Route ${idx + 1}</strong><br>
        <span class="metric">Score: ${route.score.toFixed(3)}</span><br>
        <span class="metric">Length: ${Math.round(route.length_m)} m</span>
      `;
      item.onclick = () => selectRoute(route, item);
      routesPanel.appendChild(item);
      route._element = item;
    });
  }

  function selectRoute(route, el) {
    selectedRoute = route;
    currentRoutes.forEach(r => {
      if (r._element) r._element.classList.remove('active');
    });
    if (el) el.classList.add('active');
    drawRoutes();
    renderRouteDetails(route);
  }

  function renderRouteDetails(route) {
    detailsPanel.innerHTML = '';
    detailsPanel.classList.remove('hidden');

    const title = document.createElement('h3');
    title.textContent = 'Route Details';
    detailsPanel.appendChild(title);

    const ph = document.createElement('strong');
    ph.textContent = 'Pros:';
    detailsPanel.appendChild(ph);

    const pl = document.createElement('ul');
    pl.style.marginTop = '.2rem';
    route.pros.forEach(p => {
      const li = document.createElement('li');
      li.textContent = p;
      pl.appendChild(li);
    });
    detailsPanel.appendChild(pl);

    const ch = document.createElement('strong');
    ch.textContent = 'Cons:';
    detailsPanel.appendChild(ch);

    const cl = document.createElement('ul');
    cl.style.marginTop = '.2rem';
    route.cons.forEach(c => {
      const li = document.createElement('li');
      li.textContent = c;
      cl.appendChild(li);
    });
    detailsPanel.appendChild(cl);

    const sh = document.createElement('strong');
    sh.textContent = 'Stops:';
    detailsPanel.appendChild(sh);

    const sl = document.createElement('ul');
    sl.style.marginTop = '.2rem';
    route.stops.forEach(s => {
      const li = document.createElement('li');
      li.textContent = `(${s.x}, ${s.y}) - ${s.note}`;
      sl.appendChild(li);
    });
    detailsPanel.appendChild(sl);

    vrButton.classList.remove('hidden');
    vrButton.onclick = () => startVr(route);
  }

  // --------- VR ----------
  function startVr(route) {
    // Hide 2D UI
    mapWrapper.style.display = 'none';
    routesPanel.classList.add('hidden');
    detailsPanel.classList.add('hidden');
    vrButton.classList.add('hidden');
    vrContainer.classList.remove('hidden');

    // Clear any previous scene content
    while (vrScene.firstChild) {
      vrScene.removeChild(vrScene.firstChild);
    }

    // Convert DEM to texture
    const demCanvas = demToImageData(currentTerrain.dem);
    const demUrl = demCanvas.toDataURL();

    // Scale: 1 DEM pixel -> 0.05 world units
    const width = currentTerrain.shape[1];
    const height = currentTerrain.shape[0];
    const scale = 0.05;

    // Terrain plane
    const plane = document.createElement('a-plane');
    plane.setAttribute('position', `${(width * scale) / 2} 0 ${(height * scale) / 2}`);
    plane.setAttribute('width', `${width * scale}`);
    plane.setAttribute('height', `${height * scale}`);
    plane.setAttribute('rotation', '-90 0 0');
    plane.setAttribute('material', `src: ${demUrl}; repeat: 1 1; side: double`);
    vrScene.appendChild(plane);

    // Route polyline segments (requires aframe-line-component)
    for (let i = 0; i < route.path.length - 1; i++) {
      const p1 = route.path[i];
      const p2 = route.path[i + 1];
      const seg = document.createElement('a-entity');
      // Use a single string for multi-prop set:
      seg.setAttribute(
        'line',
        `start: ${(p1.x * scale).toFixed(3)} 0.02 ${(p1.y * scale).toFixed(3)}; ` +
        `end: ${(p2.x * scale).toFixed(3)} 0.02 ${(p2.y * scale).toFixed(3)}; ` +
        `color: #00ffae`
      );
      seg.setAttribute('material', 'color: #00ffae; opacity: 0.8');
      vrScene.appendChild(seg);
    }

    // Stops (animated bouncing spheres)
    route.stops.forEach(s => {
      const sp = document.createElement('a-sphere');
      sp.setAttribute('radius', '0.05');
      sp.setAttribute('color', '#ffcc00');
      sp.setAttribute('position', `${(s.x * scale).toFixed(3)} 0.05 ${(s.y * scale).toFixed(3)}`);
      // ✅ Fixed: one clean template literal (no concatenation)
      sp.setAttribute(
        'animation',
        `property: position; dir: alternate; dur: 1500; loop: true; ` +
        `to: ${(s.x * scale).toFixed(3)} 0.1 ${(s.y * scale).toFixed(3)}`
      );
      vrScene.appendChild(sp);
    });

    // Camera rig (WASD + mouse look)
    const camRig = document.createElement('a-entity');
    camRig.setAttribute(
      'position',
      `${(route.path[0].x * scale).toFixed(3)} 1.6 ${(route.path[0].y * scale).toFixed(3)}`
    );
    const cam = document.createElement('a-camera');
    cam.setAttribute('wasd-controls-enabled', 'true');
    cam.setAttribute('look-controls', 'pointerLockEnabled:true');
    camRig.appendChild(cam);
    vrScene.appendChild(camRig);
  }

  exitVrBtn.onclick = () => {
    vrContainer.classList.add('hidden');
    mapWrapper.style.display = 'block';
    routesPanel.classList.remove('hidden');
    detailsPanel.classList.remove('hidden');
    while (vrScene.firstChild) {
      vrScene.removeChild(vrScene.firstChild);
    }
  };

  // --------- Boot ----------
  loadSuggestions().then(() => {
    modal.classList.remove('hidden');
  });

  coordForm.addEventListener('submit', ev => {
    ev.preventDefault();
    const id = +coordForm.imageId.value;
    const x = +coordForm.coordX.value;
    const y = +coordForm.coordY.value;
    if (!Number.isFinite(id) || !Number.isFinite(x) || !Number.isFinite(y)) {
      alert('Enter valid numbers');
      return;
    }
    modal.classList.add('hidden');
    computeRoutes(id, x, y);
  });

  // Optional: redraw on resize (keeps canvas sharp if window size changes)
  window.addEventListener('resize', () => {
    if (currentTerrain) drawMap(currentTerrain.dem, currentTerrain.hazards);
    if (currentRoutes.length) drawRoutes();
  });
})();

