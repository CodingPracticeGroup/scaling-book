// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-",
    title: "",
    section: "Navigation",
    handler: () => {
      window.location.href = "/scaling-book/";
    },
  },{id: "dropdown-第-0-部分-引言",
              title: "第 0 部分 · 引言",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-1-部分-roofline-入门",
              title: "第 1 部分 · Roofline 入门",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-2-部分-漫谈-tpu",
              title: "第 2 部分 · 漫谈 TPU",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-3-部分-分片矩阵乘法",
              title: "第 3 部分 · 分片矩阵乘法",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-4-部分-transformer",
              title: "第 4 部分 · Transformer",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-5-部分-训练",
              title: "第 5 部分 · 训练",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-6-部分-训练-llama",
              title: "第 6 部分 · 训练 LLaMA",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-7-部分-推理",
              title: "第 7 部分 · 推理",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-8-部分-部署-llama",
              title: "第 8 部分 · 部署 LLaMA",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-9-部分-性能剖析",
              title: "第 9 部分 · 性能剖析",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-10-部分-漫谈-jax",
              title: "第 10 部分 · 漫谈 JAX",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-11-部分-结论",
              title: "第 11 部分 · 结论",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{id: "dropdown-第-12-部分-gpu",
              title: "第 12 部分 · GPU",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "";
              },
            },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
