import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';

let scene, camera, renderer, controls, mixer, clock;
let currentModel = null;

const modelMap = {
  escuchar: '/model_escuchar.fbx',
  procesar: '/model_procesar.fbx',
  hablar: '/model_hablar.fbx',
};

let cameraLight, cameraLightTarget;

init();
animate();
loadModel('/model_escuchar.fbx'); // Carga inicial por defecto

// 📡 WebSocket con hostname dinámico
const wsHost = `${window.location.hostname}:8765`;
const socket = new WebSocket(`ws://${wsHost}`);

socket.onopen = () => {
  console.log(`✅ WebSocket conectado con el backend en ws://${wsHost}`);
};

socket.onerror = (error) => {
  console.error('❌ Error en WebSocket:', error);
};

socket.onmessage = (event) => {
  try {
    const data = JSON.parse(event.data);
    console.log('📩 Mensaje recibido del backend:', data);

    const modelo = modelMap[data.estado];
    if (modelo) {
      console.log(`🔁 Cambiando modelo a estado: "${data.estado}" → ${modelo}`);
      loadModel(modelo);
    } else {
      console.warn('⚠️ Estado desconocido recibido:', data.estado);
    }
  } catch (e) {
    console.error('❌ Error al parsear el mensaje del WebSocket:', e);
  }
};

function init() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeeeeee);

  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 2, 5);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);

  // Luz ambiental y direccional global
  const light1 = new THREE.DirectionalLight(0xffffff, 1);
  light1.position.set(5, 10, 7.5);
  scene.add(light1);
  scene.add(new THREE.AmbientLight(0x404040));

  // Luz pegada a la cámara
  cameraLight = new THREE.DirectionalLight(0xffffff, 0.8);
  cameraLight.position.set(0, 0, 0);
  camera.add(cameraLight);
  scene.add(camera);

  // Target de la luz (para que apunte al rostro)
  cameraLightTarget = new THREE.Object3D();
  scene.add(cameraLightTarget);
  cameraLight.target = cameraLightTarget;

  clock = new THREE.Clock();
  window.addEventListener('resize', onWindowResize, false);
}

function loadModel(modelPath) {
  const loader = new FBXLoader();

  // Limpieza del modelo anterior
  if (currentModel) {
    scene.remove(currentModel);
    currentModel.traverse((child) => {
      if (child.isMesh) {
        child.geometry.dispose();
        if (child.material.map) child.material.map.dispose();
        child.material.dispose();
      }
    });
  }

  loader.load(
    modelPath,
    (object) => {
      console.log('✅ Modelo cargado exitosamente:', modelPath);

      mixer = new THREE.AnimationMixer(object);
      if (object.animations.length > 0) {
        mixer.clipAction(object.animations[0]).play();
      }

      object.scale.set(0.01, 0.01, 0.01);
      scene.add(object);
      currentModel = object;

      // Ajustar cámara y luz al modelo
      requestAnimationFrame(() => {
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        const offsetZ = size.z * 2.2;
        const offsetY = size.y * 0.4;

        camera.position.set(center.x, center.y + offsetY, center.z + offsetZ);
        camera.lookAt(center.x, center.y + offsetY, center.z);

        controls.target.set(center.x, center.y + offsetY, center.z);
        controls.update();

        cameraLightTarget.position.set(center.x, center.y + offsetY, center.z);
      });
    },
    undefined,
    (error) => {
      console.error('❌ Error al cargar modelo:', modelPath, error);
    }
  );
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  if (mixer) mixer.update(delta);
  controls.update();
  renderer.render(scene, camera);
}
