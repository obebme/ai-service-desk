const replyTemplate = document.querySelector("#reply-template");
const replyBody = document.querySelector("#reply-body");

replyTemplate.addEventListener("change", () => {
  const template = replyTemplate.value;
  if (template && !replyBody.value.trim()) {
    replyBody.value = template;
    replyBody.focus();
  }
  replyTemplate.value = "";
});
