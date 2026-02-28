/** Drawing canvas component for handwriting input. */
export function createCanvas(size = 280) {
    const wrapper = document.createElement('div');

    const canvas = document.createElement('canvas');
    canvas.className = 'draw-canvas';
    canvas.width = size;
    canvas.height = size;

    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, size, size);

    let drawing = false;
    let brushSize = 12;
    let lastX = 0, lastY = 0;
    let onChangeCallback = null;

    function draw(x, y) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = brushSize;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        lastX = x;
        lastY = y;
    }

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        if (e.touches) {
            return [
                (e.touches[0].clientX - rect.left) * scaleX,
                (e.touches[0].clientY - rect.top) * scaleY,
            ];
        }
        return [
            (e.clientX - rect.left) * scaleX,
            (e.clientY - rect.top) * scaleY,
        ];
    }

    function startDraw(e) {
        e.preventDefault();
        drawing = true;
        [lastX, lastY] = getPos(e);
    }

    function moveDraw(e) {
        if (!drawing) return;
        e.preventDefault();
        const [x, y] = getPos(e);
        draw(x, y);
        lastX = x;
        lastY = y;
    }

    function stopDraw(e) {
        if (!drawing) return;
        drawing = false;
        if (onChangeCallback) onChangeCallback();
    }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', moveDraw);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseleave', stopDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', moveDraw, { passive: false });
    canvas.addEventListener('touchend', stopDraw);

    // Controls
    const controls = document.createElement('div');
    controls.className = 'flex gap-8 mt-8';
    controls.innerHTML = `
        <button class="btn" id="canvas-clear">Clear</button>
        <label class="text-sm" style="display:flex;align-items:center;gap:4px;">
            Brush: <input type="range" min="4" max="24" value="12" id="canvas-brush" style="width:80px;">
        </label>
    `;

    wrapper.appendChild(canvas);
    wrapper.appendChild(controls);

    // Wire controls after append
    setTimeout(() => {
        const clearBtn = wrapper.querySelector('#canvas-clear');
        const brushInput = wrapper.querySelector('#canvas-brush');
        if (clearBtn) clearBtn.addEventListener('click', () => {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, size, size);
            if (onChangeCallback) onChangeCallback();
        });
        if (brushInput) brushInput.addEventListener('input', (e) => {
            brushSize = parseInt(e.target.value);
        });
    }, 0);

    return {
        element: wrapper,
        canvas,
        getPixels() {
            // Downsample to 28x28 and return flat float array
            const tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = 28;
            tmpCanvas.height = 28;
            const tmpCtx = tmpCanvas.getContext('2d');
            tmpCtx.drawImage(canvas, 0, 0, 28, 28);
            const imageData = tmpCtx.getImageData(0, 0, 28, 28);
            const pixels = [];
            for (let i = 0; i < imageData.data.length; i += 4) {
                pixels.push(imageData.data[i] / 255.0); // R channel (grayscale)
            }
            return pixels;
        },
        clear() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, size, size);
        },
        onChange(cb) {
            onChangeCallback = cb;
        },
    };
}