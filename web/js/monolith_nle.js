import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function isInputLinked(node, inputName) {
    if (!node.inputs) return false;
    const inp = node.inputs.find((i) => i.name === inputName);
    return Boolean(inp && inp.link != null);
}

function getWidgetValue(node, widgetName, fallback) {
    if (!node.widgets) return fallback;
    const w = node.widgets.find((w) => w.name === widgetName);
    return w && w.value !== undefined ? w.value : fallback;
}

const TRACK_PALETTE = {
    V1: { fill: "rgba(14, 165, 233, 0.34)", stroke: "#0ea5e9", text: "#e0f2fe" },
    V2: { fill: "rgba(139, 92, 246, 0.34)", stroke: "#8b5cf6", text: "#ede9fe" },
    V3: { fill: "rgba(245, 158, 11, 0.34)", stroke: "#f59e0b", text: "#fef3c7" },
    V4: { fill: "rgba(236, 72, 153, 0.34)", stroke: "#ec4899", text: "#fce7f3" },
    A1: { fill: "rgba(16, 185, 129, 0.32)", stroke: "#10b981", text: "#d1fae5" },
};

function syncTimelineInputs(node, timelineInfo) {
    if (!node || !timelineInfo || !Array.isArray(timelineInfo.inputs)) return;

    const desired = timelineInfo.inputs;
    const desiredNames = new Set(desired.map((d) => d.name));

    // If upgrading an old node in-place that had media_video linked, rename media_video to the first video clip
    if (node.inputs && desired.length > 0) {
        const firstVideoSpec = desired.find((d) => d.role === "video") || desired[0];
        const hasFirstAlready = node.inputs.some((i) => i.name === firstVideoSpec.name);
        if (!hasFirstAlready) {
            const legacyVid = node.inputs.find((i) => i.name === "media_video" && i.link != null);
            if (legacyVid) {
                legacyVid.name = firstVideoSpec.name;
                legacyVid.type = firstVideoSpec.type;
                legacyVid.label = firstVideoSpec.name;
            }
        }
    }

    // Remove inputs that are no longer part of the timeline
    if (node.inputs) {
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            if (!desiredNames.has(inp.name)) {
                node.removeInput(i);
            }
        }
    }

    // Add or update desired inputs in exact timeline order
    for (let idx = 0; idx < desired.length; idx++) {
        const spec = desired[idx];
        let existingIdx = node.inputs ? node.inputs.findIndex((i) => i.name === spec.name) : -1;
        if (existingIdx === -1) {
            node.addInput(spec.name, spec.type);
            existingIdx = node.inputs.length - 1;
        } else {
            node.inputs[existingIdx].type = spec.type;
        }
        const slot = node.inputs[existingIdx];
        slot.label = spec.role === "matte" ? `${spec.name} [Matte]` : spec.name;
    }

    // Reorder node.inputs array to match desired order while updating link target_slot indices
    if (node.inputs && node.inputs.length === desired.length) {
        const ordered = [];
        for (const spec of desired) {
            const found = node.inputs.find((i) => i.name === spec.name);
            if (found) ordered.push(found);
        }
        if (ordered.length === desired.length) {
            node.inputs = ordered;
            if (app.graph && app.graph.links) {
                for (let slotIdx = 0; slotIdx < node.inputs.length; slotIdx++) {
                    const lid = node.inputs[slotIdx].link;
                    if (lid != null && app.graph.links[lid]) {
                        app.graph.links[lid].target_slot = slotIdx;
                    }
                }
            }
        }
    }

    if (node.setSize && node.computeSize && node.size) {
        const sz = node.computeSize();
        node.setSize([Math.max(node.size[0], sz[0], 380), Math.max(node.size[1], sz[1])]);
    }
    if (app.graph) {
        app.graph.setDirtyCanvas(true, true);
    }
}

async function fetchAndApplyTimeline(node, filename, refreshCb) {
    if (!filename || filename === "none") {
        node._monolithOtioInfo = null;
        if (refreshCb) refreshCb();
        return;
    }
    try {
        const resp = await api.fetchApi(`/monolith/nle/timeline_info?file=${encodeURIComponent(filename)}`);
        if (!resp.ok) return;
        const data = await resp.json();
        if (data && data.valid) {
            node._monolithOtioInfo = data;
            syncTimelineInputs(node, data);
            if (refreshCb) refreshCb();
        }
    } catch (err) {
        console.warn("Monolith NLE: failed to load timeline info:", err);
    }
}

function getTimelineCanvasHeight(node) {
    const info = node._monolithNleMeta || node._monolithOtioInfo;
    const tracks = info && Array.isArray(info.ui_tracks) ? info.ui_tracks : [];
    const count = Math.max(3, tracks.length);
    return 50 + count * 28;
}

function drawNleTimeline(canvas, node) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(300, rect.width || (node.size ? node.size[0] - 20 : 360));
    const height = getTimelineCanvasHeight(node);

    if (canvas.style.height !== `${height}px`) {
        canvas.style.height = `${height}px`;
    }
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
    }

    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.scale(dpr, dpr);

    const info = node._monolithNleMeta || node._monolithOtioInfo;
    const hwEnc = (node._monolithNleMeta && node._monolithNleMeta.hw_encoder) || getWidgetValue(node, "hw_encoder", "auto");
    const totalDur = info && info.total_dur ? Math.max(0.5, parseFloat(info.total_dur)) : 13.125;
    const specs = info && info.width
        ? `${info.width}x${info.height} @ ${info.fps}fps (${info.total_frames}f)`
        : "Select .otio / .xml timeline";

    // Background panel
    ctx.fillStyle = "#14171c";
    ctx.strokeStyle = "#2b313c";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(0.5, 0.5, width - 1, height - 1, 6);
    ctx.fill();
    ctx.stroke();

    // Header bar
    ctx.fillStyle = "#1d222b";
    ctx.beginPath();
    ctx.roundRect(1, 1, width - 2, 24, [6, 6, 0, 0]);
    ctx.fill();

    ctx.font = "600 10px Inter, system-ui, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("OTIO TIMELINE", 8, 16);

    const badgeText = `${specs}  |  ${totalDur.toFixed(2)}s  |  ${String(hwEnc).toUpperCase()}`;
    ctx.font = "500 10px monospace";
    ctx.fillStyle = "#38bdf8";
    const bw = ctx.measureText(badgeText).width;
    ctx.fillText(badgeText, Math.max(96, width - bw - 8), 16);

    const labelColW = 46;
    const trackX = labelColW + 6;
    const trackW = Math.max(100, width - trackX - 10);

    // Time Ruler (y = 25..42)
    ctx.fillStyle = "#181c23";
    ctx.fillRect(1, 25, width - 2, 17);
    ctx.strokeStyle = "#262c36";
    ctx.beginPath();
    ctx.moveTo(0, 42);
    ctx.lineTo(width, 42);
    ctx.stroke();

    const stepSec = totalDur <= 6 ? 1 : totalDur <= 16 ? 2 : 5;
    ctx.font = "9px monospace";
    ctx.fillStyle = "#64748b";
    for (let t = 0; t <= totalDur + 0.01; t += stepSec) {
        const x = trackX + (t / totalDur) * trackW;
        ctx.strokeStyle = "#334155";
        ctx.beginPath();
        ctx.moveTo(x, 36);
        ctx.lineTo(x, 42);
        ctx.stroke();
        ctx.fillText(`${t.toFixed(0)}s`, Math.min(width - 20, x + 2), 35);
    }

    const rawTracks = info && Array.isArray(info.ui_tracks) ? info.ui_tracks : [];
    const videoTracks = rawTracks.filter((t) => t.kind === "video").slice().reverse();
    const audioTracks = rawTracks.filter((t) => t.kind === "audio");
    const displayTracks = [...videoTracks, ...audioTracks];

    let rowY = 46;
    const rowH = 24;

    for (const tr of displayTracks) {
        // Track label cell
        ctx.fillStyle = "#1b2028";
        ctx.fillRect(4, rowY, labelColW - 4, rowH);
        ctx.font = "600 9px monospace";
        ctx.fillStyle = tr.kind === "audio" ? "#6ee7b7" : "#94a3b8";
        ctx.fillText(tr.id, 10, rowY + rowH / 2 + 3);

        // Track lane background
        ctx.fillStyle = "#101317";
        ctx.strokeStyle = "#222731";
        ctx.beginPath();
        ctx.roundRect(trackX, rowY, trackW, rowH, 4);
        ctx.fill();
        ctx.stroke();

        const pal = TRACK_PALETTE[tr.id] || (tr.kind === "audio" ? TRACK_PALETTE.A1 : TRACK_PALETTE.V2);

        for (const cl of tr.clips || []) {
            const x0 = trackX + Math.max(0, Math.min(1, cl.start / totalDur)) * trackW;
            const x1 = trackX + Math.max(0, Math.min(1, cl.end / totalDur)) * trackW;
            const cw = Math.max(10, x1 - x0);
            const linked = isInputLinked(node, cl.name) || (cl.matte && isInputLinked(node, cl.matte));

            ctx.fillStyle = linked ? "rgba(56, 189, 248, 0.42)" : pal.fill;
            ctx.strokeStyle = linked ? "#38bdf8" : pal.stroke;
            ctx.lineWidth = linked ? 1.6 : 1.1;
            ctx.beginPath();
            ctx.roundRect(x0 + 1, rowY + 2, Math.max(8, cw - 2), rowH - 4, 3);
            ctx.fill();
            ctx.stroke();

            const prefix = linked ? "[IN] " : "";
            const matteTag = cl.matte ? ` [+${cl.matte}]` : "";
            const label = `${prefix}${cl.name}${matteTag}`;

            ctx.save();
            ctx.beginPath();
            ctx.rect(x0 + 3, rowY + 2, Math.max(4, cw - 6), rowH - 4);
            ctx.clip();
            ctx.font = "500 9.5px Inter, system-ui, sans-serif";
            ctx.fillStyle = pal.text;
            ctx.fillText(label, x0 + 6, rowY + 15);
            ctx.restore();
        }

        rowY += rowH + 4;
    }

    // Playhead midpoint marker
    const thumbTs = Math.min(2.0, totalDur / 2.0);
    const thumbX = trackX + Math.min(1.0, thumbTs / totalDur) * trackW;
    ctx.strokeStyle = "#f43f5e";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 2]);
    ctx.beginPath();
    ctx.moveTo(thumbX, 38);
    ctx.lineTo(thumbX, height - 6);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.restore();
}

function attachUploadTimelineButton(node, onTimelineSelected) {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".otio,.xml";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    fileInput.onchange = async () => {
        if (!fileInput.files || !fileInput.files.length) return;
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append("image", file);
        formData.append("overwrite", "true");
        try {
            const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body: formData,
            });
            if (resp.ok) {
                const res = await resp.json();
                const uploadedName = res.name || file.name;
                const tlWidget = node.widgets ? node.widgets.find((w) => w.name === "timeline_file") : null;
                if (tlWidget) {
                    if (tlWidget.options && Array.isArray(tlWidget.options.values)) {
                        if (!tlWidget.options.values.includes(uploadedName)) {
                            tlWidget.options.values.unshift(uploadedName);
                        }
                    }
                    tlWidget.value = uploadedName;
                }
                if (onTimelineSelected) {
                    onTimelineSelected(uploadedName);
                }
            }
        } catch (err) {
            console.error("Monolith NLE: failed to upload timeline file:", err);
        }
    };

    const btn = node.addWidget("button", "Upload .otio / .xml", null, () => {
        fileInput.click();
    });
    btn.serialize = false;

    const origRemoved = node.onRemoved;
    node.onRemoved = function () {
        if (origRemoved) origRemoved.apply(this, arguments);
        if (fileInput.parentNode) {
            fileInput.parentNode.removeChild(fileInput);
        }
    };
}

function attachVideoPlayerAndTimeline(node, includeTimeline) {
    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "6px";
    container.style.width = "100%";
    container.style.boxSizing = "border-box";
    container.style.padding = "2px 0";

    let canvas = null;
    if (includeTimeline) {
        canvas = document.createElement("canvas");
        canvas.style.width = "100%";
        canvas.style.height = "190px";
        canvas.style.display = "block";
        canvas.style.borderRadius = "6px";
        container.appendChild(canvas);
    }

    const videoEl = document.createElement("video");
    videoEl.controls = true;
    videoEl.loop = true;
    videoEl.muted = true;
    videoEl.autoplay = true;
    videoEl.playsInline = true;
    videoEl.style.width = "100%";
    videoEl.style.maxHeight = "260px";
    videoEl.style.borderRadius = "6px";
    videoEl.style.background = "#090b0e";
    videoEl.style.display = "none";
    container.appendChild(videoEl);

    const domWidget = node.addDOMWidget("monolith_nle_view", "custom", container, {
        serialize: false,
        hideOnZoom: false,
    });

    domWidget.computeSize = function (width) {
        const timelineH = includeTimeline ? getTimelineCanvasHeight(node) + 8 : 0;
        const videoH = videoEl.style.display !== "none" ? 220 : 0;
        return [width, timelineH + videoH];
    };

    const refresh = () => {
        if (canvas) {
            drawNleTimeline(canvas, node);
        }
        if (node.setSize && node.computeSize && node.size) {
            const sz = node.computeSize();
            node.setSize([Math.max(node.size[0], sz[0], 380), Math.max(node.size[1], sz[1])]);
        }
    };

    attachUploadTimelineButton(node, (uploadedFile) => {
        fetchAndApplyTimeline(node, uploadedFile, refresh);
    });

    const tlWidget = node.widgets ? node.widgets.find((w) => w.name === "timeline_file") : null;
    if (tlWidget) {
        const origCb = tlWidget.callback;
        tlWidget.callback = function (val) {
            if (origCb) origCb.apply(this, arguments);
            fetchAndApplyTimeline(node, val, refresh);
        };
        setTimeout(() => {
            fetchAndApplyTimeline(node, tlWidget.value, refresh);
        }, 40);
    }

    if (includeTimeline) {
        const origConn = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            if (origConn) origConn.apply(this, arguments);
            refresh();
        };
        const origResize = node.onResize;
        node.onResize = function () {
            if (origResize) origResize.apply(this, arguments);
            if (canvas) drawNleTimeline(canvas, node);
        };
    }

    const origExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        if (origExecuted) origExecuted.apply(this, arguments);
        if (output && output.nle_timeline && output.nle_timeline[0]) {
            node._monolithNleMeta = output.nle_timeline[0];
        }
        const vidInfo = (output && output.videos && output.videos[0]) || (output && output.images && output.images[0]);
        if (vidInfo && vidInfo.filename && String(vidInfo.filename).endsWith(".mp4")) {
            const params = new URLSearchParams({
                filename: vidInfo.filename,
                subfolder: vidInfo.subfolder || "",
                type: vidInfo.type || "output",
                t: String(Date.now()),
            });
            videoEl.src = api.apiURL(`/view?${params.toString()}`);
            videoEl.style.display = "block";
        }
        refresh();
    };

    node._monolithRefreshTimeline = refresh;
}

function setWidgetVisible(w, visible) {
    if (!w) return;
    if (!Object.prototype.hasOwnProperty.call(w, "_monolithOrigComputeSize")) {
        w._monolithOrigComputeSize = w.computeSize;
    }
    w.hidden = !visible;
    if (visible) {
        if (w._monolithOrigComputeSize) {
            w.computeSize = w._monolithOrigComputeSize;
        } else {
            delete w.computeSize;
        }
    } else {
        w.computeSize = () => [0, -4];
    }
}

function setupFaceSwapSaveVideoToggle(node) {
    const saveW = node.widgets ? node.widgets.find((w) => w.name === "save_video") : null;
    if (!saveW) return;

    const prefixW = node.widgets ? node.widgets.find((w) => w.name === "filename_prefix") : null;
    const applyReplacements = window.comfyAPI?.utils?.applyTextReplacements;
    if (prefixW && typeof applyReplacements === "function") {
        prefixW.serializeValue = () => applyReplacements(app, prefixW.value);
    }

    const syncSaveVideoWidgets = () => {
        const show = Boolean(saveW.value);
        for (const name of ["filename_prefix", "video_quality", "use_gpu_encode"]) {
            const w = node.widgets ? node.widgets.find((item) => item.name === name) : null;
            setWidgetVisible(w, show);
        }
        if (node.setSize && node.computeSize && node.size) {
            const sz = node.computeSize();
            node.setSize([node.size[0], sz[1]]);
        }
        if (app.graph) {
            app.graph.setDirtyCanvas(true, true);
        }
    };

    const origCb = saveW.callback;
    saveW.callback = function () {
        const r = origCb ? origCb.apply(this, arguments) : undefined;
        syncSaveVideoWidgets();
        return r;
    };

    syncSaveVideoWidgets();
}

function upgradeInputSlotType(node, inputName, newType) {
    if (!node || !node.inputs) return;
    const inp = node.inputs.find((i) => i.name === inputName);
    if (inp) {
        inp.type = newType;
    }
}

app.registerExtension({
    name: "Monolith.NLE",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (
            nodeData.name === "MonolithNLEAssemblerV1" ||
            nodeData.name === "MonolithNLEAssemblerV2" ||
            nodeData.name === "MonolithNLEAssemblerV3"
        ) {
            const includeTimeline = nodeData.name === "MonolithNLEAssemblerV2";
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (origCreated) origCreated.apply(this, arguments);
                if (this.size) {
                    this.size[0] = Math.max(this.size[0], 390);
                }
                attachVideoPlayerAndTimeline(this, includeTimeline);
            };
            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                // Migrate legacy pre-OTIO widget array if loading an older workflow
                if (
                    info &&
                    Array.isArray(info.widgets_values) &&
                    (info.widgets_values.length > 4 || typeof info.widgets_values[1] === "boolean")
                ) {
                    const tl = typeof info.widgets_values[0] === "string" && info.widgets_values[0].endsWith(".otio")
                        ? info.widgets_values[0]
                        : "dutton_ranch_premiere.otio";
                    info.widgets_values = [tl, "auto", false];
                }
                if (origConfigure) origConfigure.apply(this, arguments);
                const hwW = this.widgets ? this.widgets.find((w) => w.name === "hw_encoder") : null;
                if (hwW && typeof hwW.value !== "string") {
                    hwW.value = "auto";
                }
                const tlFile = getWidgetValue(this, "timeline_file", "dutton_ranch_premiere.otio");
                fetchAndApplyTimeline(this, tlFile, this._monolithRefreshTimeline);
            };
        } else if (nodeData.name === "MonolithNLEOverlayTrack") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (origCreated) origCreated.apply(this, arguments);
                upgradeInputSlotType(this, "overlay_mask", "MASK,VIDEO,IMAGE");
            };
            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (origConfigure) origConfigure.apply(this, arguments);
                upgradeInputSlotType(this, "overlay_mask", "MASK,VIDEO,IMAGE");
            };
        } else if (nodeData.name === "MonolithPromptSwitchByImage") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (origCreated) origCreated.apply(this, arguments);
                upgradeInputSlotType(this, "image", "IMAGE,VIDEO");
            };
            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (origConfigure) origConfigure.apply(this, arguments);
                upgradeInputSlotType(this, "image", "IMAGE,VIDEO");
            };
        } else if (nodeData.name === "MonolithFaceSwap") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (origCreated) origCreated.apply(this, arguments);
                setupFaceSwapSaveVideoToggle(this);
            };

            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                if (
                    info &&
                    Array.isArray(info.widgets_values) &&
                    info.widgets_values.length === 19 &&
                    typeof info.widgets_values[16] === "number"
                ) {
                    info.widgets_values = [
                        ...info.widgets_values.slice(0, 16),
                        false,
                        "video/ComfyUI",
                        ...info.widgets_values.slice(16),
                    ];
                }
                if (origConfigure) origConfigure.apply(this, arguments);
                setupFaceSwapSaveVideoToggle(this);
            };
        }
    },
});
